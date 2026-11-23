"""
HeterPyramidCollabMulticlass
-----------------------------
HEAL Pyramid collaborative perception model with a CenterPoint-style
multi-class detection head compatible with V2Xverse's
center_point_loss_multiclass and VoxelPostprocessor.

Design decisions:
  - Keep the full forward_collab Pyramid fusion path (unchanged from
    heter_pyramid_collab.py) — only the detection head differs.
  - Replace anchor-based heads (cls=K, reg=7K, dir=2K) with
    CenterPoint-style heads (cls=num_class, reg=8*num_class).
  - Add generate_predicted_boxes (copied from center_point_codriving)
    to produce reg_preds_multiclass used by the postprocessor.
  - Preserve occ_single_list output so the occ supervision path can
    still be used if desired (it is simply not consumed by
    center_point_loss_multiclass, which is fine).
  - num_class is taken from args['anchor_number'] to stay consistent
    with the YAML key used by the V2Xverse multiclass configs.
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision


class HeterPyramidCollabMulticlass(nn.Module):
    """
    Pyramid collaborative perception with CenterPoint multiclass heads.

    Key differences from HeterPyramidCollab:
      * cls_head  : (in_head → num_class)        — per-class heatmap
      * reg_head  : (in_head → 8 * num_class)    — 8-param center encoding
      * no dir_head
      * forward() returns cls_preds, reg_preds_multiclass, bbox_preds
        (matching centerpointcodriving output contract expected by
         center_point_loss_multiclass and VoxelPostprocessor)
    """

    def __init__(self, args):
        super(HeterPyramidCollabMulticlass, self).__init__()
        self.args = args

        modality_name_list = list(args.keys())
        modality_name_list = [
            x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()
        ]
        self.modality_name_list = modality_name_list

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {}

        # ------------------------------------------------------------------ #
        # Per-modality: encoder / backbone / aligner
        # ------------------------------------------------------------------ #
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')
            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            setattr(self, f"encoder_{modality_name}",
                    encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            setattr(self, f"backbone_{modality_name}",
                    ResNetBEVBackbone(model_setting['backbone_args']))
            setattr(self, f"aligner_{modality_name}",
                    AlignNet(model_setting['aligner_args']))

            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}",
                        (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}",
                        (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}",
                        (camera_mask_args['grid_conf']['xbound'][1] -
                         camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}",
                        (camera_mask_args['grid_conf']['ybound'][1] -
                         camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}":
                        eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}":
                        eval(f"self.crop_ratio_H_{modality_name}"),
                }

        # ------------------------------------------------------------------ #
        # Feature-space geometry
        # ------------------------------------------------------------------ #
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        # ------------------------------------------------------------------ #
        # Pyramid fusion backbone (unchanged)
        # ------------------------------------------------------------------ #
        self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])

        # ------------------------------------------------------------------ #
        # Optional shrink header (unchanged)
        # ------------------------------------------------------------------ #
        self.shrink_flag = False
        if 'shrink_header' in args:
            self.shrink_flag = True
            self.shrink_conv = DownsampleConv(args['shrink_header'])

        # ------------------------------------------------------------------ #
        # CenterPoint-style multiclass detection heads
        #   anchor_number == num_class (3 for V2Xverse: car/ped/cyclist)
        # ------------------------------------------------------------------ #
        self.num_class = args['anchor_number']
        in_ch = args['in_head']

        self.cls_head = nn.Conv2d(in_ch, self.num_class, kernel_size=1)
        self.reg_head = nn.Conv2d(in_ch, 8 * self.num_class, kernel_size=1)

        # CenterPoint init (from centerpointcodriving.init_weight)
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.reg_head.weight, mean=0, std=0.001)

        # ------------------------------------------------------------------ #
        # Box-decode parameters (needed by generate_predicted_boxes)
        # ------------------------------------------------------------------ #
        self.voxel_size = args['voxel_size']        # [vw, vh, vd]
        self.out_size_factor = args['out_size_factor']

        # ------------------------------------------------------------------ #
        # Optional compressor
        # ------------------------------------------------------------------ #
        self.compress = False
        if 'compressor' in args:
            self.compress = True
            self.compressor = NaiveCompressor(args['compressor']['input_dim'],
                                              args['compressor']['compress_ratio'])

        self.model_train_init()
        check_trainable_module(self)

    # ---------------------------------------------------------------------- #
    # Training init (same logic as HeterPyramidCollab)
    # ---------------------------------------------------------------------- #
    def model_train_init(self):
        if self.compress:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            self.compressor.train()
            for p in self.compressor.parameters():
                p.requires_grad_(True)

    # ---------------------------------------------------------------------- #
    # Box decode (identical to centerpointcodriving.generate_predicted_boxes)
    # ---------------------------------------------------------------------- #
    def generate_predicted_boxes(self, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            cls_preds : (B, H, W) or (B, 1, H, W) — single-class heatmap
            box_preds : (B, 8, H, W) — center-point 8-param encoding

        Returns:
            cls_preds  : unchanged
            batch_box_preds : (B, H*W, 7)  [x, y, z, h, w, l, rot]
        """
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        batch, H, W, code_size = box_preds.size()
        box_preds = box_preds.reshape(batch, H * W, code_size)

        batch_reg = box_preds[..., 0:2]

        h = box_preds[..., 3:4] * self.out_size_factor * self.voxel_size[0]
        w = box_preds[..., 4:5] * self.out_size_factor * self.voxel_size[1]
        l = box_preds[..., 5:6] * self.out_size_factor * self.voxel_size[2]
        batch_dim = torch.cat([h, w, l], dim=-1)
        batch_hei = (box_preds[..., 2:3] * self.out_size_factor * self.voxel_size[2]
                     + self.cav_range[2])

        batch_rots = box_preds[..., 6:7]
        batch_rotc = box_preds[..., 7:8]
        rot = torch.atan2(batch_rots, batch_rotc)

        ys, xs = torch.meshgrid(
            [torch.arange(0, H), torch.arange(0, W)], indexing='ij'
        )
        ys = ys.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)
        xs = xs.view(1, H, W).repeat(batch, 1, 1).to(cls_preds.device)

        xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
        ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.cav_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.cav_range[1]

        batch_box_preds = torch.cat([xs, ys, batch_hei, batch_dim, rot], dim=2)
        return cls_preds, batch_box_preds

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #
    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab'}

        record_len = data_dict['record_len']

        # ------------------------------------------------------------------
        # V2Xverse compatibility: agent_modality_list is absent in the
        # non-heterogeneous V2Xverse pipeline (all agents are LiDAR/m1).
        # Synthesise it from record_len when missing.
        # ------------------------------------------------------------------
        if 'agent_modality_list' in data_dict:
            agent_modality_list = data_dict['agent_modality_list']
        else:
            n_total = int(record_len.sum())
            agent_modality_list = ['m1'] * n_total

        # ------------------------------------------------------------------
        # V2Xverse compatibility: the batch stores voxel data under
        # 'processed_lidar' (matching centerpointcodriving's convention),
        # but heter_encoders.PointPillar.forward() looks for
        # data_dict['inputs_{modality_name}'].  Bridge by aliasing the key
        # for each lidar modality that is missing its 'inputs_*' entry.
        # We do NOT mutate processed_lidar itself — only add the alias key.
        # ------------------------------------------------------------------
        for modality_name in self.modality_name_list:
            inputs_key = f'inputs_{modality_name}'
            if inputs_key not in data_dict:
                if self.sensor_type_dict[modality_name] == 'lidar':
                    if 'processed_lidar' in data_dict:
                        data_dict[inputs_key] = data_dict['processed_lidar']

        affine_matrix = normalize_pairwise_tfm(
            data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size
        )
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        # ---- per-modality encode ----------------------------------------- #
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")(
                {"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

        # ---- camera crop/pad (skipped for LiDAR-only V2Xverse) ----------- #
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H * eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W * eval(f"self.crop_ratio_W_{modality_name}"))
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict[f"depth_items_{modality_name}"] = \
                            eval(f"self.encoder_{modality_name}").depth_items

        # ---- assemble heterogeneous feature stack ------------------------- #
        counting_dict = {mn: 0 for mn in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(
                modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        heter_feature_2d = torch.stack(heter_feature_2d_list)

        if self.compress:
            heter_feature_2d = self.compressor(heter_feature_2d)

        # ---- Pyramid fusion ---------------------------------------------- #
        fused_feature, occ_outputs = self.pyramid_backbone.forward_collab(
            heter_feature_2d,
            record_len,
            affine_matrix,
            agent_modality_list,
            self.cam_crop_info,
        )

        if self.shrink_flag:
            fused_feature = self.shrink_conv(fused_feature)

        # ---- CenterPoint multiclass heads --------------------------------- #
        cls = self.cls_head(fused_feature)   # (B, num_class, H, W)
        bbox = self.reg_head(fused_feature)  # (B, 8*num_class, H, W)

        # Decode per-class boxes: produce reg_preds_multiclass (B, num_class, H*W, 7)
        box_preds_for_infer = bbox.permute(0, 2, 3, 1).contiguous()
        # reshape to (B, H, W, num_class, 8)
        B, H_f, W_f, _ = box_preds_for_infer.shape
        box_preds_for_infer = box_preds_for_infer.view(
            B, H_f, W_f, self.num_class, 8
        )

        bbox_temp_list = []
        for i in range(self.num_class):
            # (B, 8, H, W) for class i
            bp_i = box_preds_for_infer[:, :, :, i, :].permute(0, 3, 1, 2)
            _, bbox_temp = self.generate_predicted_boxes(cls[:, i, :, :], bp_i)
            bbox_temp_list.append(bbox_temp)          # each (B, H*W, 7)
        reg_preds_multiclass = torch.stack(bbox_temp_list, dim=1)  # (B, num_class, H*W, 7)

        # Single-class decode (for backward compat with postprocessor
        # that may look at reg_preds / cls_preds keys)
        _, reg_preds_flat = self.generate_predicted_boxes(cls, bbox)

        output_dict.update({
            'cls_preds': cls,                              # (B, num_class, H, W)
            'reg_preds': reg_preds_flat,                   # (B, H*W, 7)
            'reg_preds_multiclass': reg_preds_multiclass,  # (B, num_class, H*W, 7)
            'bbox_preds': bbox,                            # (B, 8*num_class, H, W) — raw, needed by loss
            'occ_single_list': occ_outputs,
            # fused_feature required by pnp_infer_action_e2e.py line 874:
            #   fused_feature_2 = perception_results['fused_feature'].permute(0,1,3,2)
            #   feature = fused_feature_3[:,:,:192,:]   → (B, 128, 192, 96)
            # Our fused_feature is (B, 128, 96, 288); after permute(0,1,3,2) → (B,128,288,96)
            # after flip dim2 → (B,128,288,96); slice [:,:,:192,:] → (B,128,192,96) ✓
            'fused_feature': fused_feature,               # (B, 128, 96, 288)
        })

        return output_dict
