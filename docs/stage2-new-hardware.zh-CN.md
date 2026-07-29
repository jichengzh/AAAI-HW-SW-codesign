# Stage2 新硬件接入边界

新硬件首先进入 Stage1，而不是作为 Stage2 的独立参数进入。

## 正确路径

1. 编写硬件 capability YAML。
2. 运行 Stage1 scan，把硬件能力归一化进 manifest。
3. Stage2 读取 manifest 中的 `hw_capability`、`round_to`、`int8_buildable_align` 等字段。
4. 若有真实测量，在 Stage2/TVM 路径中产生 `stage2_evidence_delta_v1`。
5. Stage1 evidence / calibrated predictor / model classifier 吸收 delta 后重新生成 gate。

```bash
PYTHONPATH=. python -m framework.stage1.run_scan \
  --model <registry_name> \
  --hw configs/hardware/<new_target>.yaml \
  --device cuda \
  --out-dir framework/partitions
```

随后运行 Stage2：

```bash
PYTHONPATH=. python scripts/stage2_optimize_model.py \
  --manifest framework/partitions/<registry_name>_partition.yaml \
  --classification results/stage1_model_predict/model_classifier/stage1_model_classification_v1.json \
  --out-json results/stage2/<registry_name>/stage2_optimization_v1.json
```

## Stage2 不接收的参数

Stage2 公共入口不提供：

- `hardware_target`
- `evidence_registry_path`
- `search_policy`

原因是这些信息已经有明确归属：

- 硬件 capability 由 Stage1 scan 写入 manifest。
- 原始 evidence 由 Stage1 evidence / classifier 或 Stage2 evidence delta 链路管理。
- 搜索策略由 classifier gate 和模型级 Stage2 search space 派生。

## 能声明什么

静态 capability 能支持：

- 合法位宽和粒度判断。
- `round_to` / `int8_buildable_align` 等搜索空间收缩。
- 哪些候选在结构上可搜索。

静态 capability 不能支持：

- 新硬件真实 latency 结论。
- 新硬件 AP/accuracy 结论。
- full-model speedup claim。

若只测了 RSU dense-core，就只能声明 dense-core 范围内的优化结果。ego fusion、attention、routing、sparse VFE、postprocess 等未测子图必须保留为 future block、blocker 或 coverage caveat。
