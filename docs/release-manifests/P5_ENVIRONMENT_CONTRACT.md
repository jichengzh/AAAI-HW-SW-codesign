# P5 环境契约

P5 只提供 CPU reference、RTX 4090 与 H800 的公开、脱敏环境契约和纯离线验证入口。
硬件 YAML（`configs/hardware/*.yaml`）是能力权威：它声明目标平台的架构、精度和功能。
环境 YAML（`configs/environment/*.yaml`）是运行时约束：它声明 Python、框架、CUDA 与驱动版本范围，
并引用相应硬件能力文件。观测 JSON 是用户显式提供的、已脱敏的本地输入；验证器不会自行收集或
推断任何观测值。

## 离线验证

调用者须自行准备与目标环境 YAML 对应的脱敏观测 JSON，然后传入三个参数：

```bash
python tools/release/validate_environment_contract.py \
  --contract configs/environment/rtx4090.yaml \
  --observation <redacted-observation.json> \
  --output <environment-contract-report.json>
```

`validate_environment_contract.py` 读取 CLI 指定的 environment contract 和 observation，写出 output，
并验证 environment contract 引用的公开 hardware capability YAML。它是纯离线工具：
不探测 GPU，不读取环境变量，不访问网络或读取外部资源，不运行 CUDA 或 CUDA 编译，不运行训练、
评测或基准，且不下载数据、权重、工具链或任何其他资产。

## 稳定结果

- 退出码 `0`：输入有效，且观测满足环境契约。
- 退出码 `1`：输入有效，但观测不满足环境契约。
- 退出码 `2`：参数、契约、观测或报告写入无效；工具尽力写出固定的脱敏无效报告。

报告只包含目标、通过状态和稳定失败代码；不会回显观测内容或文件系统细节。此契约仅记录离线
配置兼容性，不构成真实硬件可用性、真实 GPU 观测、CUDA 执行、训练/评测/基准结果或论文性能证据。

## 阶段边界

P5 当前仍为**进行中（本地）**。真实设备访问、CUDA 执行、训练、评测、基准、外部资产获取和
性能主张均不在本契约范围内，须由后续阶段在获得明确授权后另行处理。
