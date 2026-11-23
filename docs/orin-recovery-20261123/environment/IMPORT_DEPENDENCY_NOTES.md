# UniAD环境最小导入依赖补充

2026-11-23核对两份Python版本TSV、已保存的MMCV/MMDetection `requirements/runtime.txt`，并通过只读SSH检查实际源文件的导入语句及已安装发行包的`Requires-Dist`元数据。**没有在远端安装或运行推理，也没有在新机器执行安装/导入验证。**本次只补两套UniAD运行子集，不构建完整训练依赖闭包。

## 为何补充这些包

| 来源/触发 | 必须准备的包 | 本次记录的版本 |
|---|---|---|
| MMCV1.5/1.6 `utils/config.py` | addict、yapf | 两环境均2.4.0、0.40.1 |
| MMCV顶层`__init__`导入image/video/visualization；`image/io.py`直接`import cv2` | OpenCV、numpy、Pillow、matplotlib | venv：4.6.0.66/1.22.4/9.5.0/3.5.1；conda：4.12.0.88/1.22.4/10.4.0/3.5.3 |
| MMCV `requirements/runtime.txt` | packaging、PyYAML等 | 原运行子集已有packaging；venv继承系统PyYAML5.3.1，conda为6.0.2 |
| MMDetection `requirements/runtime.txt`；`mmdet/core/evaluation/mean_ap.py`等 | matplotlib、pycocotools、six、terminaltables | pycocotools2.0.7、terminaltables3.1.10；six在venv1.14.0、conda1.17.0 |
| yapf0.40.1发行元数据 | importlib-metadata≥6.6、platformdirs≥3.5.1、tomli≥2 | 两环境importlib-metadata8.5.0、platformdirs4.3.6；tomli分别2.4.1/2.2.1 |
| matplotlib实际依赖与导入路径 | fonttools、python-dateutil及resolver选出的兼容cycler/kiwisolver/pyparsing | 补已观测fonttools4.57.0与dateutil2.9.0.post0；其余可由包依赖解析安装/系统继承 |
| MM系列部分表格/评估入口 | prettytable | 两环境均3.11.0 |

所有新增pin均在**对应环境自己的TSV**中出现，包含venv继承系统的six；新增19组不同包名/版本还逐项核对PyPI官方`requires_python`，均声明允许Python3.8.10，这仅是解释器版本范围检查，不验证依赖闭包或ABI。没有把当前上游最新版本写成原机器版本。PyPI名称中的`importlib_metadata`与`importlib-metadata`等按规范化名称视为同包，venv还看到系统旧metadata1.5.0，不能用它代替虚拟环境实际安装的8.5.0。

两份源`requirements/runtime.txt`没有完整声明所有顶层可导入模块；例如`cv2`是实际代码的直接导入。因此`pip install --no-deps -e .`只安装MMCV/MMDet自身不足以保证导入，须先安装这些基础包。

## OpenCV与numpy：按Python版本判断

远端已装OpenCV4.12.0.88的**实际发行元数据**和[PyPI官方元数据](https://pypi.org/pypi/opencv-python/4.12.0.88/json)声明：

```text
numpy<2.0; python_version < "3.9"
numpy<2.3.0,>=2; python_version >= "3.9"
```

所以原conda **Python3.8.19 + OpenCV4.12.0.88 + numpy1.22.4在声明约束上兼容**。本次保留这三个已观测版本；这不等于已验证二进制ABI或图像读写。如果迁到Python3.9+，同一OpenCV版本会要求numpy2，不能继续与numpy1.22.4拼成“可安装闭包”。迁移应另建环境，或在新环境验证已观测的OpenCV4.6.0.66等较早版本及其wheel；不要静默升级numpy并沿用Torch1.12/MMCV旧扩展。

venv的OpenCV4.6.0.66在Linux aarch64 Python3.8上声明numpy≥1.19.3，numpy1.22.4满足这一项。[PyPI4.6.0.66元数据](https://pypi.org/pypi/opencv-python/4.6.0.66/json)

一个环境只选择一种pip OpenCV发行包，避免同时装`opencv-python`、`opencv-contrib-python`和headless变体；venv继承的apt cv2也需检查`cv2.__file__`，不能把系统OpenCV和pip版本的元数据混为一谈。宿主Jetson/OpenCV开发库是另一层，别为修复某个venv而删除系统库。

## 新系统的安装与验收顺序

按主环境文档先固定Python3.8、系统库和Torch。venv还需通过apt恢复`python3-yaml`（当前5.3.1）且使用`--system-site-packages`；本次未在pip子集加入旧PyYAML5.3.1以免把apt状态误写成必须从源码构建的pip包。conda使用子集中自己的PyYAML6.0.2。

```bash
export RECOVERY_ENV=/home/jichengzhi/orin-recovery/github/environment
# 二选一：使用当前正在恢复的Python，不能混用。
P=/home/jichengzhi/uniad_venv/bin/python
REQ="$RECOVERY_ENV/requirements_uniad_venv_runtime.txt"
# P=/home/jichengzhi/miniforge3/envs/uniad/bin/python
# REQ="$RECOVERY_ENV/requirements_miniforge_uniad_runtime.txt"

# 主文档中的pip install -r必须允许解析第三方包依赖，不能给这一行加--no-deps。
"$P" -m pip install -r "$REQ"
# 在相应源码目录按主文档重编MMCV ops、安装MMDetection；其--no-deps以此基础依赖已到位为前提。
"$P" -m pip check
```

`pip check`发现其他旧训练包冲突时，按实际要运行的路径处理并记录；不要批量升级整套Torch/MM系列。本清单仍保留原Numba/llvmlite、NuScenes等历史pin，但不声称它们的所有数值API与训练扩展都通过兼容验证。

若aarch64没有可用pycocotools wheel而需要源码编译，应先装固定numpy，再让扩展针对它编译，避免构建隔离环境意外使用numpy2。下面是**尚未新机验证的备用构建方案**，不是本次安装记录：

```bash
# 先确保主环境文档的build-essential、python3-dev已安装。
"$P" -m pip install numpy==1.22.4 Cython==3.2.8
# Cython3.2.8在原venv可观测；用于conda重建时属于待验证构建工具选择。
"$P" -m pip install --no-build-isolation --no-deps --no-binary=pycocotools pycocotools==2.0.7
"$P" -m pip install -r "$REQ"
```

第三方图像包也可能需系统`libGL`等动态库；依具体导入错误补系统依赖，不以pip包版本存在判断动态库已齐全。

下面只做导入/配置API检查，**不加载checkpoint、不执行模型推理、不连接底盘**。在新系统安装并重编MMCV之后手动执行；本次未运行：

```bash
"$P" -B - <<'PY'
import cv2, numpy, matplotlib, addict, yapf, terminaltables, pycocotools
import mmcv, mmdet
from mmcv import Config
from mmcv.ops import RoIPool
print('cv2', cv2.__version__, cv2.__file__)
print('numpy', numpy.__version__, numpy.__file__)
print('mmcv/mmdet', mmcv.__version__, mmdet.__version__)
print(Config(dict(a=1)).pretty_text)
PY
```

`Config.pretty_text`会触发源码中的`FormatCode(..., verify=True)`；原yapf0.40.1已被明确固定，避免未验证升级造成旧API失效。MMCV ops导入通过仍不证明所有算子/导出/模型正确。完整UniAD项目、mmdet3d编译和训练依赖按对应源码入口另外验收，不在此次最小补充范围内。
