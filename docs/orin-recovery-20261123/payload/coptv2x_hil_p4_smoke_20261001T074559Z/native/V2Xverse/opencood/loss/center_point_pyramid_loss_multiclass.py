# -*- coding: utf-8 -*-
"""
CenterPointPyramidLossMulticlass
---------------------------------
Hybrid loss = CenterPoint multiclass detection loss
            + HEAL Pyramid per-agent occupancy (occ) supervision.

Detection part  : identical logic to CenterPointLossmulticlass
                  (cls Gaussian focal + reg L1, per class)
Occ part        : sigmoid focal loss on each level of occ_single_list
                  produced by PyramidFusion.forward_collab.
                  Occupancy GT is derived from the CenterPoint heatmaps:
                  union of all-class heatmaps downsampled to each pyramid
                  level's resolution.

Why not reuse calc_occ_loss from PointPillarPyramidLoss?
  Because that method needs pos_equal_one / neg_equal_one (anchor-grid
  format), which the V2Xverse intermediatemulticlass dataset does NOT
  produce.  The multiclass dataset only provides object_bbx_center /
  object_bbx_mask (centerpoint format).  We therefore derive occupancy
  from the already-computed CenterPoint heatmaps.

occ_single_list layout (from PyramidFusion.forward_collab):
  List of length num_levels, each element shape (N_total, 1, H_i, W_i)
  where N_total = sum(record_len) across the batch (all CAVs stacked).
  For occ supervision we only supervise the ego agent of each sample
  (index 0 of each group in record_len).  This matches HEAL's intention:
  single-agent occ heads teach each CAV to detect occupancy on its own.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reuse helpers from existing losses — import by module path so we never
# modify those files.
from opencood.loss.center_point_loss_multiclass import (
    CenterPointLossmulticlass,
    clip_sigmoid,
    gaussian_focal_loss as _gaussian_focal_loss_decorated,
)
from opencood.loss.point_pillar_loss import sigmoid_focal_loss


class CenterPointPyramidLossMulticlass(nn.Module):
    """
    Args (from yaml loss.args):
      cls_weight         : float  — weight for classification heatmap loss
      loc_weight         : float  — weight for regression loss
      code_weights       : list   — per-dim weights for 8-param box loss
      occ_weight         : float  — global weight applied to total occ loss
      occ_cls_args       : dict   — {alpha, gamma, type} for occ focal loss
                                    defaults: alpha=0.25, gamma=2.0
      occ_pos_cls_weight : float  — positive sample weight in occ focal loss
                                    default: 2.0
      pyramid            : dict   — {relative_downsample: [...], weight: [...]}
      target_assigner_config : dict — passed through to CenterPointLossmulticlass
    """

    def __init__(self, args):
        super(CenterPointPyramidLossMulticlass, self).__init__()

        # ---- Detection loss (delegate to existing class) -----------------
        # Build a CenterPointLossmulticlass instance that we call internally.
        self._det_loss = CenterPointLossmulticlass(args)

        # ---- Occ loss parameters -----------------------------------------
        self.occ_weight = args.get('occ_weight', 1.0)
        occ_cls_args = args.get('occ_cls_args', {})
        self.occ_alpha = occ_cls_args.get('alpha', 0.25)
        self.occ_gamma = occ_cls_args.get('gamma', 2.0)
        self.occ_pos_cls_weight = args.get('occ_pos_cls_weight', 2.0)

        pyramid_cfg = args.get('pyramid', {
            'relative_downsample': [1, 2, 4],
            'weight': [0.5, 0.35, 0.2],
        })
        self.relative_downsample = pyramid_cfg['relative_downsample']
        self.pyramid_weight = pyramid_cfg['weight']
        self.num_levels = len(self.relative_downsample)

        # Unified loss dict exposed for logging
        self.loss_dict = {}

    # ---------------------------------------------------------------------- #
    # Main entry point
    # ---------------------------------------------------------------------- #
    def forward(self, output_dict, target_dict, suffix=""):
        """
        Parameters
        ----------
        output_dict : dict
            Must contain:
              cls_preds        (B, num_class, H, W)
              bbox_preds       (B, 8*num_class, H, W)
              occ_single_list  list[(N_total, 1, H_i, W_i)] × num_levels
        target_dict : dict
            Must contain:
              object_bbx_center{suffix}  (B, num_class, max_obj, 7)
              object_bbx_mask{suffix}    (B, num_class, max_obj)
        """
        # ---- 1. Detection loss ------------------------------------------- #
        det_loss = self._det_loss(output_dict, target_dict, suffix)

        # ---- 2. Occ supervision ------------------------------------------ #
        occ_loss = torch.tensor(0.0, device=det_loss.device)
        occ_single_list = output_dict.get('occ_single_list', [])

        if len(occ_single_list) > 0:
            occ_loss = self._calc_occ_loss(
                occ_single_list=occ_single_list,
                output_dict=output_dict,
                target_dict=target_dict,
                suffix=suffix,
            )

        total_loss = det_loss + self.occ_weight * occ_loss

        self.loss_dict = {
            'total_loss': total_loss.item(),
            'det_loss':   det_loss.item(),
            'occ_loss':   occ_loss.item() if isinstance(occ_loss, torch.Tensor) else occ_loss,
        }
        # Merge inner detection loss dict for logging
        self.loss_dict.update(self._det_loss.loss_dict)

        return total_loss

    # ---------------------------------------------------------------------- #
    # Occ loss calculation
    # ---------------------------------------------------------------------- #
    def _calc_occ_loss(self, occ_single_list, output_dict, target_dict, suffix):
        """
        Derive per-level occupancy GT from CenterPoint heatmaps and supervise
        each level of occ_single_list with sigmoid focal loss.

        occ_single_list[i] shape : (N_total, 1, H_i, W_i)
          N_total = sum(record_len) — all CAVs in the batch stacked.
          We extract the ego row of each batch sample (index 0 within
          each sample's group) so we train each agent's occ head.

        Occupancy GT at finest resolution = union of all-class heatmaps
          cls_preds shape : (B, num_class, H, W)
          We use the GT heatmaps from the target assigner (recomputed
          here at low cost) rather than cls_preds, so the supervision
          signal is label-driven, not prediction-driven.

        In the interest of keeping this self-contained without re-running
        the full target assigner, we build a binary occupancy map from
        object_bbx_center / object_bbx_mask:
          For each batch item, project 3D box centres onto the BEV grid,
          set those cells to 1 → gives a sparse binary (not Gaussian)
          foreground mask.  This is coarser but correct for focal loss.
        """
        record_len = output_dict.get('record_len',
                     target_dict.get('record_len', None))
        if record_len is None:
            # Cannot extract ego slice — skip occ loss gracefully
            return torch.tensor(0.0)

        batch_size = len(record_len)
        device = occ_single_list[0].device

        # ---- Extract ego-only rows from N_total -------------------------
        # record_len[b] = number of CAVs in sample b.  Ego is row 0.
        ego_indices = []
        offset = 0
        for b in range(batch_size):
            ego_indices.append(offset)
            offset += int(record_len[b])
        # occ_ego_list[i] : (B, 1, H_i, W_i)
        occ_ego_list = [
            occ_single_list[i][ego_indices]
            for i in range(len(occ_single_list))
        ]

        # ---- Build binary occupancy GT at finest occ level resolution ----
        # Use the heatmap resolution from the first occ level.
        _, _, H_occ0, W_occ0 = occ_ego_list[0].shape  # e.g. (B,1,H_i,W_i)

        # object_bbx_center{suffix}: (B, num_class, max_obj, 7) or (B, max_obj, 7)
        bbx_key = f'object_bbx_center{suffix}'
        msk_key = f'object_bbx_mask{suffix}'
        if bbx_key not in target_dict:
            # Fallback: no occ GT available
            return torch.tensor(0.0, device=device)

        bbx_center = target_dict[bbx_key]  # may be (B, K, N, 7) or (B, N, 7)
        bbx_mask   = target_dict[msk_key]

        # Flatten class dimension if present → (B, K*N, 7) then union
        if bbx_center.ndim == 4:
            B, K, N, _ = bbx_center.shape
            bbx_flat = bbx_center.view(B, K * N, 7)
            msk_flat = bbx_mask.view(B, K * N)
        else:
            bbx_flat = bbx_center  # (B, N, 7)
            msk_flat = bbx_mask

        # Get lidar range and voxel size from target_assigner_config
        target_cfg = self._det_loss.target_cfg
        pc_range   = self._det_loss.lidar_range   # [xmin, ymin, zmin, xmax, ymax, zmax]
        voxel_size = self._det_loss.voxel_size     # [vw, vh, vd]
        out_factor = target_cfg['out_size_factor']

        # Build binary foreground map at finest occ resolution (H_occ0, W_occ0)
        # These are PyramidFusion's per-level resolutions:
        #   level 0: stride 1 relative to fused BEV (same as fused feature)
        # The fused feature after shrink is (B, 128, 96, 288) for this config.
        # occ_ego_list[0] has H_i = ? — use its actual shape, not H_occ0.
        # For each level we build the GT by downsampling from a "full" resolution.

        # Full BEV grid (before any occ downsampling):
        # x: [pc_range[0], pc_range[3]], step = voxel_size[0]
        # grid_x = (pc_range[3]-pc_range[0]) / voxel_size[0] = 576 → /out_factor=2 → 288
        # grid_y = (pc_range[4]-pc_range[1]) / voxel_size[1] = 192 → /out_factor=2 → 96
        # This matches H=96, W=288 from P2.
        # For occ: each level i has relative_downsample[i] = 1,2,4
        # → occ level 0: 96×288 (stride 1)
        #   occ level 1: 48×144 (stride 2)
        #   occ level 2: 24×72  (stride 4)

        # Build binary GT at base resolution = occ level 0 resolution
        H0 = int(occ_ego_list[0].shape[2])
        W0 = int(occ_ego_list[0].shape[3])

        # Build foreground binary map [B, H0, W0]
        occ_gt_base = torch.zeros(batch_size, H0, W0, device=device)
        for b in range(batch_size):
            valid = msk_flat[b].bool()  # (N,) or (K*N,)
            boxes = bbx_flat[b][valid]  # (n_valid, 7): x,y,z,h,w,l,yaw
            if boxes.shape[0] == 0:
                continue
            # Project (x, y) to occ grid indices
            cx = boxes[:, 0]
            cy = boxes[:, 1]
            ix = ((cx - pc_range[0]) / voxel_size[0] / out_factor
                  * H0 / (  # scale from full-grid to occ0 grid
                    (pc_range[3] - pc_range[0]) / voxel_size[0] / out_factor
                  )).long()
            iy = ((cy - pc_range[1]) / voxel_size[1] / out_factor
                  * W0 / (
                    (pc_range[4] - pc_range[1]) / voxel_size[1] / out_factor
                  )).long()
            # Clamp to valid range
            ix = ix.clamp(0, H0 - 1)
            iy = iy.clamp(0, W0 - 1)
            occ_gt_base[b, ix, iy] = 1.0

        # ---- Compute occ loss per level ----------------------------------
        total_occ_loss = torch.tensor(0.0, device=device)
        for i in range(min(len(occ_ego_list), self.num_levels)):
            occ_pred = occ_ego_list[i]        # (B, 1, H_i, W_i)
            ds = self.relative_downsample[i]

            # Downsample GT using max-pool to match this level's resolution
            if ds > 1:
                occ_gt_level = F.max_pool2d(
                    occ_gt_base.unsqueeze(1).float(),  # (B,1,H0,W0)
                    kernel_size=ds, stride=ds
                )  # (B, 1, H_i, W_i)
            else:
                occ_gt_level = occ_gt_base.unsqueeze(1).float()  # (B,1,H0,W0)

            # Flatten: (B, H_i*W_i, 1)
            B_cur, _, H_i, W_i = occ_pred.shape
            occ_pred_flat = occ_pred.permute(0, 2, 3, 1).contiguous().view(B_cur, -1, 1)
            occ_gt_flat   = occ_gt_level.permute(0, 2, 3, 1).contiguous().view(B_cur, -1, 1)

            positives  = (occ_gt_flat > 0).float()
            negatives  = (occ_gt_flat == 0).float()
            pos_norm   = positives.sum(1, keepdim=True).float()

            occ_weights = (positives * self.occ_pos_cls_weight
                           + negatives * 1.0)
            occ_weights /= torch.clamp(pos_norm, min=1.0)

            level_loss = sigmoid_focal_loss(
                occ_pred_flat,
                occ_gt_flat,
                weights=occ_weights,
                alpha=self.occ_alpha,
                gamma=self.occ_gamma,
            )
            level_loss = level_loss.sum() / batch_size
            level_loss = level_loss * self.pyramid_weight[i]
            total_occ_loss = total_occ_loss + level_loss

        return total_occ_loss

    # ---------------------------------------------------------------------- #
    # Logging
    # ---------------------------------------------------------------------- #
    def logging(self, epoch, batch_id, batch_len, writer=None, suffix=""):
        total_loss = self.loss_dict.get('total_loss', 0)
        det_loss   = self.loss_dict.get('det_loss', 0)
        occ_loss   = self.loss_dict.get('occ_loss', 0)
        cls_loss   = self.loss_dict.get('cls_loss', 0)
        reg_loss   = self.loss_dict.get('reg_loss', 0)

        print(
            "[epoch %d][%d/%d]%s || Total: %.4f || Det: %.4f "
            "|| Conf: %.4f || Loc: %.4f || Occ: %.4f"
            % (epoch, batch_id + 1, batch_len, suffix,
               total_loss, det_loss, cls_loss, reg_loss, occ_loss)
        )

        if writer is not None:
            writer.add_scalar('Total_loss' + suffix,
                              total_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Det_loss' + suffix,
                              det_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Occ_loss' + suffix,
                              occ_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Regression_loss' + suffix,
                              reg_loss, epoch * batch_len + batch_id)
            writer.add_scalar('Confidence_loss' + suffix,
                              cls_loss, epoch * batch_len + batch_id)
