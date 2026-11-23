# 原路径与恢复来源映射

`payload/` 与离线资产均按远端 `/home/jichengzhi/` 的相对路径布局。固定原家目录能减少脚本、editable包和配置的路径改动；它不自动补全被省略的数据、Git LFS对象或软链接目标。文件是否真正收录以 `manifests/` 的逐文件清单为准。

## 恢复包的组成

| 恢复材料 | 放在哪里 | 如何使用 |
|---|---|---|
| Git内 `payload/` | 本恢复仓库 | `rsync -a payload/ /home/jichengzhi/`，保留原源码/配置相对路径；不使用 `--delete` |
| Git内 `environment/`、说明和 `manifests/` | 留在恢复仓库 | 作为版本、源文件、安装步骤与完整性账本；不是要直接覆盖 `/etc` 的系统文件 |
| Git内 `key_results/` | 留在恢复仓库 | 最小原始结果/审计记录；子路径仍相对原home，源绝对路径见结果manifest；无需自动回填旧运行目录 |
| Git外 `offline_assets/critical-assets.tar.xz.part000` 等 | 和仓库分开保存 | 合并后解包，去掉第一层打包前缀，恢复到 `/home/jichengzhi/<原relative_path>` |
| Git外 `fcooper-assets.tar.gz.part*` / `supplemental-assets.tar.gz.part*` | 和仓库分开保存 | 额外定制FCooper模型、ResNet候选及UniAD单帧，同样按原home路径解包 |
| Git外 `offline_assets/ros2-foxy-images.tar.gz` | 和仓库分开保存 | `gzip -dc ... | docker load`，恢复镜像而非复制到普通目录 |

具体解包命令见 [RESTORE_CHECKLIST.md](RESTORE_CHECKLIST.md)。源码完整性、原commit/patch和链接记录见 `manifests/source_capture_manifest.json`；离线包、结果的实际文件范围以最终对应manifest为准。选择列表或文档提到一个原路径，不表示该目录全部内容已打包。

## 系统与环境

| 原绝对路径 | 内容/版本 | 恢复方式 |
|---|---|---|
| `/usr/local/cuda-11.4`、`/usr/local/cuda` | CUDA11.4，nvcc11.4.315；后者为版本链接 | 官方R35.6.5/Jetson安装源重装，由包管理维护；不从payload复制 |
| `/usr/lib/aarch64-linux-gnu` | Jetson cuDNN8.6、TRT8.5.2、驱动动态库 | 匹配BSP/apt组件重装，不套x86或SBSA包 |
| `/usr/src/tensorrt` | TensorRT工具/示例；头文件实际在`/usr/include/aarch64-linux-gnu` | 对应Jetson系统包重装 |
| `/home/jichengzhi/tensorrt-8.5.2`（新建兼容入口） | include/lib链接到Jetson apt的实际头文件/库 | 按SOURCE_PAYLOAD.md创建两个链接供旧CMake使用，不是重复安装SDK |
| `/usr/bin/python3` | Ubuntu20.04系统Python3.8.10 | 随OS/apt恢复，供ROS1及系统TRT绑定使用 |
| `/opt/ros/noetic` | 宿主ROS1 Noetic | apt重装；与Foxy容器分开 |
| `/home/jichengzhi/uniad_venv` | Python3.8.10、Torch1.12、MMCV1.5/mmdet2.25.1 | 重建venv，继承系统site-packages，按版本账本重新安装和编译 |
| `/home/jichengzhi/miniforge3` | Miniforge base，Python3.12.10、conda25.3.0 | Linux-aarch64安装器重装；用于ORT r02解释器 |
| `/home/jichengzhi/miniforge3/envs/uniad` | Python3.8.19、Torch1.12、MMCV1.6/mmdet2.26/mmdet3d1.0.0rc6 | conda specs+pip子集+源重编；原conda CUDA11.1.1不代表宿主版本 |
| `/home/jichengzhi/mmcv_build/mmcv` | v1.5.0 editable源，commit235c0253… | 从payload/源码锁恢复后在venv重编CUDA算子 |
| `/home/jichengzhi/mmcv_build/mmdetection` | v2.25.1 editable源，commit3b72b12f… | 从payload/源码锁恢复后重新editable安装 |
| `/home/jichengzhi/mmcv` | v1.6.0 editable源，commitf4167fe1… | 从payload/源码锁恢复后在conda环境重编 |
| `/home/jichengzhi/.local/lib/python3.8/site-packages/torchvision-0.13.0a0+da3794e-py3.8-linux-aarch64.egg` | 历史用户级torchvision，原两套UniAD都曾引用 | 只在定向资产清单存在时恢复该egg；推荐按源码锁在每套环境分别重编，不复制整个`.local` |
| `/home/jichengzhi/torch-1.12.0a0+2c916ef.nv22.3-cp38-cp38-linux_aarch64.whl` | 原NVIDIA Torch1.12安装wheel | 离线资产回原路径，再分别安装至所需Python3.8环境 |

各组件精确版本、wheel/架构限制与新安装命令见 [ENVIRONMENT_RESTORE.md](environment/ENVIRONMENT_RESTORE.md)。恢复包不复制旧venv/conda目录、大缓存或凭据。

## ROS1 / UniAD / HIL源码与定制资产

| 原绝对路径 | 主要用途 | 恢复来源与检查 |
|---|---|---|
| `/home/jichengzhi/ROS1/wheeltec_robot/src` | 已修复/裁剪厂商catkin包及`uniad_bridge` | payload中已收录源码；重新rosdep/catkin编译，检查源清单，不复制旧build/devel |
| `/home/jichengzhi/ROS1/uniad_integration` | ROS1与UniAD UDP整合脚本 | payload；先阅读脚本，`uniad_closedloop.sh`含底盘/控制动作，不作为安装检查执行 |
| `/home/jichengzhi/UniAD` | Python UniAD源码、配置、导出工具 | payload收录范围以源码manifest为准；完整原仓库HEAD未采到，不用当前上游自动覆盖备份配置 |
| `/home/jichengzhi/liruozhan/toOrinV2/DL4AGX_min/uniad-trt/inference_app/enqueueV2` | 实际C++推理程序及plugin源 | payload源码；按CMake重新编译，原脚本要求此工作目录及相对图片路径 |
| `/home/jichengzhi/liruozhan/toOrinV2/onnx` | 实际部署tiny FP/INT8/QDQ图 | 离线资产中保留的定制ONNX；公开dummy图无法替代 |
| `/home/jichengzhi/liruozhan/toOrinV2/uniad_tiny_fp16.engine` | ROS1历史部署引擎 | 离线资产；同栈仍需验收，TRT版本变更则按原设置重建 |
| `/home/jichengzhi/liruozhan/toOrinV2/calib_data_fixed_1150.npz` | 定制量化校准输入 | 离线资产；与构建配置、ONNX和plugin配套 |
| `/home/jichengzhi/liruozhan/toOrinV2/uniad_trt_input_119plus`、`uniad_trt_input_240` 等 | 定制选帧的TRT输入及图片引用 | 不默认恢复全部大输入；按数据文档与实际资产清单补齐，不能把小smoke输入视为整套选帧 |
| `/home/jichengzhi/ditl_rsu_edge_v1` | DITL服务、TRT backend、ROS2 bridge、Dockerfile | payload脚本/配置；依赖torch/numpy/TRT以及项目协议模块；ROS2部分在Foxy容器 |
| `/home/jichengzhi/coptv2x_hil_serial_opt_stage2_20261009/runtime` | HIL worker、服务与协议栈 | payload收录部分；恢复整套互相导入的模块，连接原host侧仿真/数据另行检查 |
| `/home/jichengzhi/coptv2x_hil_pipeline_20261010` | Pipeline去重重放与ROS记录 | payload收录脚本/配置，key_results中保存关键原始结果；不默认恢复大重放输入缓存 |
| `/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/input/preflight.npy` | 单个预录BEV输入 | 离线资产；只用于相应smoke，不含原139GiB缓存 |
| `/home/jichengzhi/coptv2x_hil_p4_recovery_20261003T054210Z` | P4恢复联调审计/五臂记录 | key_results最小原始记录及manifest；不是整套失败attempt/运行缓存 |

## Pyramid / CoDriving / FCooper / ResNet

| 原绝对路径 | 内容 | 恢复方式 |
|---|---|---|
| `/home/jichengzhi/heal_research/HEAL` | HEAL模型/数据adapter源码、配置 | payload中已收录源码/补丁；依赖及自定义算子按原版本重建；数据链接按manifest单独处理 |
| `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_pruned75_2026_05_10` | 定制pruned75 checkpoint/config | checkpoint来自离线资产、配置来自payload；不是公共基础权重 |
| `/home/jichengzhi/m4_8_orin/dair_ckpt/Pyramid_DAIR_m1_base_2023_08_14_11_42_29` | Pyramid基础checkpoint/config | 离线资产/配置，公开HEAL包也有基础权重；仍核对准确版本 |
| `/home/jichengzhi/m4_8_orin/p03_build/{base,pruned50,pruned75}.onnx` | 原Pyramid图 | 离线资产，重建对应引擎；关键旧CSV另存key_results |
| `/home/jichengzhi/lane_c_orin_stage6_five_config_20260724/models/{pyramid,codriving}` | 多宽度定制checkpoint/config | 离线模型+payload配置，具体宽度与文件以manifest为准 |
| `/home/jichengzhi/lane_c_orin_stage6_five_config_20260724/source/{pyramid,codriving}` | 各宽度ONNX/导出或构建入口 | 离线图+payload源码；不能跨宽度共用checkpoint |
| `/home/jichengzhi/lane_c_orin_fcooper_stage6_five_config_20260724` | FCooper五臂配置/构建与AP入口 | payload中已收录部分；完整OPV2V另行下载；正式表格见key_results下`V2X/results/...` |
| `/home/jichengzhi/resnet_compare_s1_20261013/isolated_torch_compat_r01` | 独立Torch2.1和10个安装wheel | 恢复wheels后重建`venv/`，不复制原venv；Python3.8、不继承系统包 |
| `/home/jichengzhi/resnet_compare_s1_20261013/ort_cpu_20261026_r01/{wheels,python}` | ORT1.19.2 cp38 CPU | wheel回原位置，用pip`--target`重装`python/`；无独立解释器 |
| `/home/jichengzhi/resnet_compare_s1_20261013/ort127_cp312` | ORT1.27/numpy等cp312 wheel集合 | 离线安装资产；不是可直接source的venv |
| `/home/jichengzhi/resnet_compare_s1_20261013/ort_cpu_20261026_r02/python` | ORT1.27目标目录 | 用Miniforge Python3.12及前行wheel重建；不要加入Python3.8路径 |
| `/home/jichengzhi/resnet_compare_s1_20261013/classification_target_accuracy_r01/model.pt` | 指定分类导出模型 | 离线资产；文件存在不代表已有准确率结论 |
| `/home/jichengzhi/resnet_compare_s1_20261013/{quant_measurement_actual_r01,target_fullval_r02,cpu_ort_full50000_*}` | 关键原始结果/审计记录 | key_results保留指定小文件；模型和完整输入缓存不是随结果自动恢复 |

## 需要重新下载或外部提供的数据

| 原/建议目标路径 | 恢复要求 |
|---|---|
| `/home/jichengzhi/lane_c_orin_backbone_parity_20260723/dair_val_full/cooperative-vehicle-infrastructure` | 原完整DAIR验证子集；本项目split1789条、两侧各1789个PCD；按数据文档重下/筛选，不恢复旧大缓存 |
| `/home/jichengzhi/m4_8_orin/dair_val/cooperative-vehicle-infrastructure` | 历史仅60对smoke子集；HEAL原链接曾指向这里，不能用来验收完整AP1789 |
| `/home/jichengzhi/heal_research/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure` | 原数据软链接；目标关系由source manifest记录，先补数据再核对链接与最终loader |
| `/data/lsw-data/OPV2V/test` | 原完整OPV2V test，2170时间戳；`/data`不在home payload恢复范围，需单独建立存储/下载 |
| `/home/jichengzhi/lane_c_orin_fcooper_stage6_five_config_20260724/data/OPV2V/test` | 历史实验内较小导出子集；不能凭目录名代替上一行完整test |
| `/home/jichengzhi/UniAD/data/{nuscenes,infos,others}` | nuScenes合法来源+匹配UniAD元数据；实际tiny部署输入仍按原选帧流程生成/另存 |
| `/data/imagenet` | **建议新目标，并非已核实原路径**；ImageNet val50000张及devkit，或恢复原host流式供数；完整样本清单不能替代图片 |

下载链接、规模口径、许可与可用性边界统一见 [DATASETS_AND_WEIGHTS.md](DATASETS_AND_WEIGHTS.md)。本恢复包只保留关键结果和必要定制资产，不承诺恢复所有原实验、数据缓存、历史尝试、逐帧输出或训练状态。
