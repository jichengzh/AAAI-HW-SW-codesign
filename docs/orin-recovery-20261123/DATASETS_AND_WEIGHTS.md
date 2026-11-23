# Orin 恢复：数据集与权重下载来源

核验日期：2026-11-23。所有路径均为格式化前 Orin 的绝对路径。此文区分 **官方标称/官方文件元数据**、**当前本地子集实测**、**未核实**；下载包大小、文件逻辑大小和磁盘占用不能混用。`null` 表示未核实/未提供，不能解释成零。可机器读取的记录见 [download_sources.json](manifests/download_sources.json)。

恢复关键结论：DAIR 的 1789 条和 OPV2V 的 2170 条是本项目评测入口，下载完整数据包后仍须恢复原 split/config。Pyramid 基础预训练权重有公开来源；`pruned75`、本项目剪枝/恢复训练权重、UniAD 实际部署 ONNX 及校准输入需自备或离线备份。下载上游基础模型不能复原这些自定义产物。离线备份清单及 SHA256 以本恢复目录其他 manifests/payload 为准；本页没有个人 Release URL 的条目，不代表已上传。

## 1. DAIR-V2X-C：协同 LiDAR 验证

官网：[DAIR-V2X](https://thudair.baai.ac.cn/index)；官方代码与当前下载入口：[AIR-THU/DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X#dataset-download-)。官方 README 将全系列总量标为 71,254 帧图像和 71,254 帧点云，**不能把这个总数当作 DAIR-V2X-C 验证对数**。完整镜像由官方链接到 [Public-V2X-Datasets](https://drive.google.com/drive/folders/1gnrw5llXAIxuB9sEKKCm6xTaJ5HQAw2e)。

直接进入：[DAIR-V2X-C / Full Dataset (train&val)](https://drive.google.com/drive/folders/1I9Du70tQ-6s0qIXfYsIHd7uBW--BwaFJ)。2026-11-23 匿名读取官方 Drive 目录所得包名和字节数如下；未下载完整包校验内容，未发现官方 SHA256。名称中的 `deprecated` 按原样保留。

| 官方文件名 | 官方目录字节数 | 文件入口 |
| --- | ---: | --- |
| `cooperative-vehicle-infrastructure-infrastructure-side-image-deprecated.zip` | 2,702,940,835 | [下载页](https://drive.google.com/file/d/1YUtOl6U-ia-HcH6VwZHRDXzIja362VXA/view) |
| `cooperative-vehicle-infrastructure-infrastructure-side-velodyne.z01` | 5,368,709,120 | [下载页](https://drive.google.com/file/d/1I23V_4fwgc0Hbl7azwlb0IblnaNmw9fw/view) |
| `cooperative-vehicle-infrastructure-infrastructure-side-velodyne.zip` | 3,505,618,661 | [下载页](https://drive.google.com/file/d/1dkC1_vJqu7vfyCfzsO3EdIeTZ1ZR81Xf/view) |
| `cooperative-vehicle-infrastructure-vehicle-side-image.zip` | 2,715,899,794 | [下载页](https://drive.google.com/file/d/1au6f6W4JeWZ6Sf-eAXQgHol6myYCsoxz/view) |
| `cooperative-vehicle-infrastructure-vehicle-side-velodyne.z01` | 5,368,709,120 | [下载页](https://drive.google.com/file/d/1vxSTXLg5f_gllfQY_vEADnCtl12ADe7n/view) |
| `cooperative-vehicle-infrastructure-vehicle-side-velodyne.z02` | 5,368,709,120 | [下载页](https://drive.google.com/file/d/18sByFR1W-4edKhRmvZVD7krjZfjbLr2I/view) |
| `cooperative-vehicle-infrastructure-vehicle-side-velodyne.zip` | 2,077,546,493 | [下载页](https://drive.google.com/file/d/1ZL9ixo0NfKJAnbKbViJwYyHBZYJr-T7z/view) |
| `cooperative-vehicle-infrastructure.zip` | 245,642,216 | [下载页](https://drive.google.com/file/d/1shLOKhwW1urQUrlqOeUtGkBl5tML_NuL/view) |

仅 LiDAR 恢复需 metadata 包 `cooperative-vehicle-infrastructure.zip` 与两侧全部 velodyne 分卷，共 **21,934,934,730 B（20.43 GiB，压缩下载量）**。这是上游 `train&val` 包集，不是现有 Orin 的 val-only 导出；全量解压大小和解压后精确帧数本次未核实。图像另计，纯 LiDAR AP 不必下载图像。该镜像目录未显示独立 test 包，不推断其覆盖所有隐藏测试数据。

分卷 ZIP 的 `.z01/.z02/.zip` 应全部保留原文件名并放同一目录，用支持分卷 ZIP 的 7-Zip 从 `.zip` 主卷解压；不要把它误当 OpenCOOD 的 `zip.part*`。先解 metadata，再合并两侧点云目录。检查解压包内部层级后，将最终树放到：

```text
/home/jichengzhi/lane_c_orin_backbone_parity_20260723/dair_val_full/
└── cooperative-vehicle-infrastructure/
    ├── cooperative/{data_info.json,label_world/}
    ├── infrastructure-side/{calib/,velodyne/}
    ├── vehicle-side/{calib/,velodyne/}
    ├── val.json
    └── train.json
```

上面的 `val.json/train.json` 和 complemented labels 应恢复本项目副本。本恢复包额外保存原完整val目录的14,315份JSON（116,821,737 B，含split/校准/标签；没有PCD或图像），放在独立 `dataset-metadata.tar.gz.part*` 中，精确SHA256见 `dataset_metadata_assets_manifest.json`。官方完整包/补充标注解压完成后，最后再回放这份metadata，避免被新包覆盖，用于恢复旧评测口径。该快照对应现有val_full子集，不能替代完整训练集metadata。HEAL 要求补充标注，入口为 [DAIR-V2X-C Complemented Annotations](https://siheng-chen.github.io/dataset/dair-v2x-c-complemented/)，直接文件：[DAIR-V2X-C_Complemented_Anno.zip](https://drive.google.com/file/d/13g3APNeHBVjPcF-nTuUoNOSGyTzdfnUK/view)。下载页本次只返回登录入口，大小/SHA 未核实。按该页面将补充的 vehicle-side `label/lidar` 和 cooperative `label_world` 放回对应位置；保留原标签备份。该页面说明其训练 split 省略了缺失路端文件的 `002234`。

**本地实测（只读扫描，不是官方下载规格）：**

| 现有路径/内容 | 实测数据量 | 实测大小 | 恢复含义 |
| --- | --- | --- | --- |
| `lane_c_orin_backbone_parity_20260723/dair_val_full/cooperative-vehicle-infrastructure/` | `val.json` 1789 条；3578 个 PCD（车/路各 1789）；17893 个普通文件 | 逐文件逻辑总和 3,540,316,091 B = 3.297 GiB；`du -s -B1` 占用 3,586,179,072 B = **3.34 GiB** | 当前完整 LiDAR 验证子集，重下载全量后按本项目清单筛选 |
| `m4_8_orin/dair_val/cooperative-vehicle-infrastructure/` | 120 个 PCD，只覆盖 60 对；`val.json` 却仍列 1789 条 | 543 个文件，逻辑总和 118,128,808 B | 旧 smoke 子集，不能恢复完整 AP1789 |
| `heal_research/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure` | 软链指向上面的 `m4_8_orin/dair_val/...` | 不重复计数据量 | 恢复原路径后仍需对完整 AP 配置核查实际 data_dir |

另一个版本差异必须保留：2026-11-23 读取[当前官方 split JSON](https://github.com/AIR-THU/DAIR-V2X/blob/c885c54af0c34bc515fa9ca8b5e8fda76a15462c/data/split_datas/cooperative-split-data.json)，其 `cooperative_split` 为 train=4813、val=1783、test=2688；本项目本地 `val.json` 是 **1789**、`train.json` 是 4811。这里只报告各自实测，未核实差异成因。不要下载当前 upstream split 直接覆盖论文实验 split。

访问/许可：官方代码 Apache-2.0 不自动等于原始数据再分发许可；按官网/下载时的数据条款使用。Drive 目录本次可匿名列文件，大文件下载可能遇到配额、登录或病毒扫描确认；未验证无账号完整下载。官网在本次抓取中不可达，可用官方 README 链接的镜像；没有编造无需认证的字节直链。

## 2. OPV2V：F-Cooper 评测输入与公开权重

官网：[UCLA Mobility Lab OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/)；[OpenCOOD 官方仓库](https://github.com/DerrickXuNu/OpenCOOD) 当前把数据托管在 [UCLA Box](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V)。官网概述为 73 个场景、约 12K 帧、约 230K 3D 框；这是全数据概述，不是本机 test 数量。

本次匿名读取 Box 公开页面元数据可确定：

| 文件/目录 | 字节数（官方 Box 页面） | 下载入口与用途 |
| --- | ---: | --- |
| `test-012.zip` | 21,146,700,611 B（19.70 GiB） | [文件页](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621736219709)，默认 towns test；应解压后核对是否包含完整所需场景 |
| `test-001.zip` | 3,413,572,752 B | [文件页](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621653380925)，同目录另一个 test 命名包；与上一包覆盖关系未核实，不自动合并或当作同一文件分卷 |
| `validate-002.zip` | 39,830,534,538 B | [文件页](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621863671123)，训练期验证，不能因配置字段名 validate 就替代 test |
| `train-003.zip` | 71,587,932,785 B | [文件页](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621828681754)，训练包；同目录还有 `train-002.zip`，是否都需要按目录说明核查 |
| `additional-001.zip` | 2,846,985,146 B | [文件页](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621920078208)，HEAL 相机模态补充数据；纯 LiDAR F-Cooper 不作为必要项 |
| `pointpillar_fcooper.zip` | **28,367,476 B** | [官方 model zoo 指定文件](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/file/1621123814293)，PointPillar + intermediate fusion，OpenCOOD/spconv1.2.1 权重 |

若大 ZIP 下载困难，官方另有 [DATA_test 分块目录](https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V/folder/280279480010)。Box 当前里面还有再次打包的 `test_chunks-20240808T...zip`；先按实际内容解出原始 `.part*`，只有见到 `test.zip.part*` 后才使用 OpenCOOD 文档的 `cat test.zip.part* > test.zip`。不要直接拼接彼此独立的 Box ZIP 文件。

**本地完整 test 实测**：`/data/lsw-data/OPV2V/test/` 共 16 个场景，5985 个多车 PCD、15940 个文件，文件逻辑总和 **20,801,228,752 B = 19.373 GiB**；每场景取固定首个 ego 目录计时间戳共 **2170**。PCD 数量包含多 agent，不能当成评测样本数。恢复到原位置并确保 `test/<scenario>/<agent>/<timestamp>.pcd` 与 `.yaml` 配对。训练时另恢复 `/data/lsw-data/OPV2V/{train,validate}`，两者当前本机完整性未核实。

**本地运行配置差异**：`/home/jichengzhi/lane_c_orin_fcooper_stage6_five_config_20260724/configs/original.yaml` 的 `root_dir=/data/lsw-data/OPV2V/train`、`validate_dir=/data/lsw-data/OPV2V/validate`，但 **`test_dir` 指向该实验目录内的 `data/OPV2V/test`**；后者只测得 647 个 PCD、2063 个文件、2,404,350,518 B（2.239 GiB），不是外部完整 test 的直接同义路径。恢复 AP2170 必须看执行脚本是否覆盖配置，并核对 loader 最终 `len(dataset)==2170`。不要仅凭字段名或目录名宣称完整验证已恢复。

权重下载后解到单独检查目录，保留包内 `config.yaml` 与 checkpoint，再按使用的 OpenCOOD/HEAL adapter 设置模型目录。**上游 F-Cooper 权重与本项目 HEAL 参数名、剪枝形状或恢复训练权重不保证兼容**。本项目定制权重须从独立离线资产压缩包恢复到原实验 `source/`/checkpoint 路径。HEAL 也公开了 [HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10.zip](https://huggingface.co/yifanlu/HEAL/resolve/main/HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10.zip)（364,092,576 B；LFS SHA256 `2459ec2894376f11d1d7ed9dcd18e348dec3308137b3c8db085310f01bc66658`），仍须匹配对应 config，不能作为自定义剪枝权重替身。

认证/许可：Box 公开列表和 F-Cooper 文件页本次可匿名读取；完整下载未执行，配额/地区/登录限制未核实。OpenCOOD 软件 MIT 许可不自动覆盖原始数据；使用/再分发遵循数据提供方条款。上述 Box 文件未发现官方 SHA256。

## 3. HEAL / Pyramid：基础 checkpoint 可下载，pruned75 必须自备

[HEAL 官方 README](https://github.com/yifanlu0227/HEAL#benchmark-checkpoints) 将 checkpoints 指向作者 [Hugging Face Hub](https://huggingface.co/yifanlu/HEAL/tree/main)。已匿名 Range 读取 `HEAL_DAIR.zip` 的 ZIP 中央目录，确认存在：

```text
stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/config.yaml
stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth
```

基础 checkpoint 未压缩 **22,122,629 B**，与 Orin 对应文件大小相同；本次未逐字节比较两者。公开整包：[HEAL_DAIR.zip 直接下载](https://huggingface.co/yifanlu/HEAL/resolve/main/HEAL_DAIR.zip)，**695,768,091 B（663.54 MiB）**；[作者仓库 API](https://huggingface.co/api/models/yifanlu/HEAL/tree/main?recursive=true&expand=false) 的 LFS SHA256：`6ddb688355433e59ef9eed7d4f323b44f9bc861300a1f8726f3708a8aa772a28`。该匿名下载支持 Range；未发现需要申请访问，独立模型许可未在元数据中确认。

下载后先检查 SHA256 与 `unzip -l HEAL_DAIR.zip`，可将 stage1 基础目录恢复到 `/home/jichengzhi/m4_8_orin/dair_ckpt/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/`，或 `/home/jichengzhi/lane_c_orin_original_fp32_20260723/ap_source/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/`。配置中绝对路径要用本项目备份副本核对。官方训练示例还支持在 HEAL 下 `opencood/logs/` 使用 checkpoint。

本地定制 **`/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10/net_epoch_bestval_at31.pth` = 8,435,411 B**，连同该目录 `config.yaml` **必须离线备份/自备**。未在官方 HEAL 文件列表找到这个文件，未获得个人 Release 链接。其它压缩比例、CoDriving 及搜索获胜点也不能由基础 zip 恢复；从离线资产压缩包取回原始checkpoint/ONNX，从payload取回配套config、模型定义与导出脚本。TensorRT engine 受软件栈/硬件影响，通常应从已保存 ONNX 在目标 Orin 重建。

## 4. UniAD / DL4AGX：base 与实际 tiny 部署分开恢复

公开实现：[OpenDriveLab/UniAD](https://github.com/OpenDriveLab/UniAD)；NVIDIA 部署：[DL4AGX / AV-Solutions/uniad-trt](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt)。NVIDIA 的 [train_export.md](https://github.com/NVIDIA/DL4AGX/blob/9f7b29104c253d5bc68334e7b83b3eecb72d4572/AV-Solutions/uniad-trt/documents/train_export.md) 中 tiny 使用 ResNet-50、BEV 50×50、输入 400×256；原 UniAD base 为 ResNet-101、BEV 200×200、输入 1600×928。两者 checkpoint 不可混用。

**公开 base 与辅助数据**（GitHub Release API 实测字节大小，未公布 digest；无需账号列出公开资产）：

| 文件 | 发布版本与直接下载 | 字节数 | 放置路径 |
| --- | --- | ---: | --- |
| `uniad_base_track_map.pth` | [v1.0](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/uniad_base_track_map.pth) | 869,687,055 | `/home/jichengzhi/UniAD/ckpts/` |
| `uniad_base_e2e.pth` | [v1.0.1](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth) | 996,345,538 | `/home/jichengzhi/UniAD/ckpts/` |
| `nuscenes_infos_temporal_train.pkl` | [v1.0](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl) | 1,033,042,168 | UniAD 项目 `data/infos/` |
| `nuscenes_infos_temporal_val.pkl` | [v1.0](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl) | 207,319,503 | UniAD 项目 `data/infos/` |
| `motion_anchor_infos_mode6.pkl` | [v1.0](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/motion_anchor_infos_mode6.pkl) | 5,109 | UniAD 项目 `data/others/` |

v1.0 也有同名 `uniad_base_e2e.pth`，大小 998,200,061 B；选版本时连同对应 config/commit 固定，不能以同名文件覆盖混用。上述 base 权重有公开入口，但不证明能重建本项目 tiny。

**tiny 的公开边界**：NVIDIA 明确只提供随机权重 [uniad_tiny_dummy.onnx](https://github.com/NVIDIA/DL4AGX/blob/master/AV-Solutions/uniad-trt/onnx/uniad_tiny_dummy.onnx)，真实模型应按文档训练得到 `tiny_imgx0.25_e2e_ep20.pth`，放在部署项目 `UniAD/ckpts/`。Orin 的 `DL4AGX/AV-Solutions/uniad-trt/onnx/uniad_tiny_dummy.onnx` 实际只有 **134 B 的 Git LFS 指针**；它声明对象大小 **302,602,872 B**，LFS SHA256 `3f849b4a45179e31a1537b8bde8a83d643a414b22fa644a775f68c989f700648`。即使 `git lfs pull` 下载了对象，也只是官方随机权重 dummy。

真实 tiny checkpoint 没有在本次核验的 NVIDIA 公共文档找到下载入口；需原作者/实验持有人提供，或自行按 tiny 两阶段配置训练。官方 ONNX 导出命令（在按 DL4AGX 文档打 patch 并安装好依赖的部署项目中运行）：

```bash
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh \
  ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py \
  ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1
```

本项目需恢复的真实部署文件在 `/home/jichengzhi/liruozhan/toOrinV2/onnx/`，**没有已核实公共下载 URL**：

| 本地文件名 | 本地实测字节数 |
| --- | ---: |
| `uniad_tiny_imgx0.25_cp.noaten_simp.onnx` | 333,254,116 |
| `uniad_tiny_imgx0.25_cp.noaten.int8.onnx` | 296,859,591 |
| `uniad_tiny_imgx0.25_cp.noaten.int8_qdq.onnx` | 396,898,227 |
| `uniad_tiny_imgx0.25_cp.noaten.int8_qdq_bb.onnx` | 396,282,407 |
| `uniad_tiny_imgx0.25_cp.noaten.int8_qdq_cnn.onnx` | 396,653,541 |
| `uniad_tiny_imgx0.25_cp.noaten.int8_qdq_noRC.onnx` | 396,835,474 |

这些 ONNX、校准输入、engine 构建脚本及 TensorRT plugin 配套恢复；本页仅确认文件存在和大小，未执行模型精度验证。真实 `.pth` 若另存于原训练 host，也应备份，其路径本次未确认。个人 Release 若后续发布，需把永久 URL 和 payload SHA256 补进 manifest；当前不能依赖它。

nuScenes 的官方拆分由 [nuScenes devkit `splits.py`](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py) 明确规定：**总计 1000 个 scene，train=700、val=150、test=150；mini 是其中 10 个 scene（mini_train=8、mini_val=2），用于调试**。因此 `v1.0-trainval` 对应 850 个场景，mini 不能替代完整验证。这里是官方场景数，不是本项目自定义选帧数。

| 官方数据版本 | 官方场景数 | 文件大小与入口 | 恢复用途 |
| --- | ---: | --- | --- |
| `v1.0-mini` | 10（8 train / 2 val） | [v1.0-mini.tgz 官方直链](https://www.nuscenes.org/data/v1.0-mini.tgz)；2026-11-23 匿名 **HEAD=200，Content-Length=4,167,696,325 B（4.168 GB / 3.881 GiB）**，仅查响应头，未下载大包 | devkit/数据预处理 smoke；不能替代论文原始选帧 |
| `v1.0-trainval` | 850（700 train / 150 val） | [官方下载页面](https://www.nuscenes.org/nuscenes#download)，需要 metadata 和相应 sensor blob 包；整套精确字节数本次未核实 | 重新训练或重新生成完整训练/验证输入 |
| `v1.0-test` | 150 | [官方下载页面](https://www.nuscenes.org/nuscenes#download)；精确包大小/字节直链本次未核实 | 官方测试任务，非当前部署恢复必需项 |

mini 的对象头没有提供 SHA256；返回的 ETag 带分段后缀，不能当作普通整文件 MD5。trainval/test 需 nuScenes 账号并遵守下载条款；mini 本次匿名 HEAD 可访问，不代表免除许可义务。需要从原始数据重做 UniAD 输入时，按 [UniAD DATA_PREP](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md) 准备部署项目 `data/nuscenes/{samples,sweeps,maps,can_bus,v1.0-trainval}`；下载后解压各数据包到同一 `data/nuscenes/`，按项目要求另补 map/CAN bus 扩展。mini 则解到同一数据根下的 `v1.0-mini/` 等目录，运行时显式选择 mini 版本。

NVIDIA [data_prep.md](https://github.com/NVIDIA/DL4AGX/blob/master/AV-Solutions/uniad-trt/documents/data_prep.md) 说明 `tools/process_metadata.py --num_frame NUM_FRAME` 生成 `UniAD/nuscenes_np/uniad_{onnx,trt}_input`，示例默认 69、允许 6..6018；本项目 `uniad_trt_input_240` 等是自定义选帧，不应以示例 69 帧替代。**仅恢复 Orin 已保存的单帧/选帧部署推理时，取回实际 ONNX、该帧图像与配套元数据/状态张量、校准输入即可，不需要下载整套训练数据**；时序推理仍须配齐其历史状态和实际帧序列。只有重训、重做原始数据预处理或完整精度评测时才需相应原始 split。

## 5. ImageNet-1K 验证集（50000）与 ResNet 权重

ImageNet 官方入口：[ILSVRC2012 下载](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)、[申请原始图像访问](https://image-net.org/request)。官方申请页要求真实姓名/机构和机构邮箱，说明不批准 gmail/hotmail 等免费邮箱请求。需要注册、按条款申请；本次没有使用任何账号或下载图像。官方站点抓取不稳定，不能保证历史直链匿名可用。

需下载文件名 **`ILSVRC2012_img_val.tar`**，如果用 `torchvision.datasets.ImageNet` 还需要 **`ILSVRC2012_devkit_t12.tar.gz`**。[TorchVision 官方实现](https://github.com/pytorch/vision/blob/main/torchvision/datasets/imagenet.py) 要求手动下载，并给出 MD5：val=`29b22e2961454d5413ddabcf34fc5622`，devkit=`fa75699e90414af021442c21a62c3abf`；未核实原始官方 SHA256。验证集 **50000 张**由 [TensorFlow Datasets 官方目录](https://www.tensorflow.org/datasets/catalog/imagenet2012) 确认。大小可参考 [Intel 官方准备说明](https://github.com/intel/ai-reference-models/blob/main/datasets/imagenet/README.md) 的 **约 6.3GB（原文标称单位）** 和 devkit 2.5MB；这不是本机实测，也不是本次从 ImageNet 官方 HEAD 核验的精确字节数。

历史源站路径 `https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar` 未在本次成功验证完整下载；请在合法账号获准后从官方页面取得当时有效链接，不能将该路径视为绕过认证的承诺。

建议单独恢复到 `/data/imagenet/`（这是本次建议的新目标路径，**不是声称原机已存在**）：

```text
/data/imagenet/
├── ILSVRC2012_img_val.tar
├── ILSVRC2012_devkit_t12.tar.gz
└── val/<WNID>/ILSVRC2012_val_*.JPEG
```

用已安装版本的 `torchvision.datasets.ImageNet('/data/imagenet', split='val')` 按 devkit 整理分类；验证 `len(dataset)==50000`。只 `tar xf` 得到平铺 JPEG 时还不满足 `ImageFolder` 类目录输入。预留原 tar 与解压图片同时存在的空间；解压后精确大小此次未核实。

本次没有确认 Orin 有完整的 50000 张原始图片。已有全量清单/CPU 流式验证结果不等于图片已在本机；原实验可依赖 host 流式输入，恢复时需重新接上原 host 或从合法源提供同一验证集。

ResNet50 基础权重可从 [TorchVision 官方 ResNet50 页面](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) 对应 `IMAGENET1K_V1` 获得；本项目已见 `resnet50-0676ba61.pth`，源站 [download.pytorch.org](https://download.pytorch.org/models/resnet50-0676ba61.pth)。本地文件 102,530,333 B；完整 SHA256 本页未计算，`0676ba61` 仅是文件名中的哈希前缀。本项目 ResNet 搜索后的 candidate ONNX 属定制资产，仍需离线备份，不能以该基础 checkpoint 复原。

## 6. 恢复验收与缺失发布项

1. 先用本恢复包 manifests 的 SHA256 验证离线自定义权重、ONNX、config；再下载上游可公开重建的内容。不要把 Git LFS 指针当模型。
2. DAIR 完整验证入口应同时满足本项目 `val.json=1789`、两侧各 1789 个目标 PCD、正确 complemented annotations；旧 60 对 smoke 子集不足以验收。
3. OPV2V 最终 loader 必须报 2170，确认实际打开的是完整 test；实验内 2.239GiB 的导出目录不足以凭大小证明全量。
4. ImageNet 以合法源和 50000 样本 loader 验收；host 流式模式还需恢复相同清单及标签映射。
5. **仍需个人 Release/离线来源**：Pyramid pruned75 与其它定制 checkpoint、CoDriving 定制权重、F-Cooper 本项目恢复训练/剪枝权重、UniAD 六个真实部署 ONNX/真实 tiny `.pth`（若存在）与实际校准/选帧输入。当前没有可核实个人公共链接，不能仅靠本页完成这些资产重建。
