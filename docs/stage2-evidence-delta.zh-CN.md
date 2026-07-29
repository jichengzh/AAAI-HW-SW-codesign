# Stage2 Evidence Delta 格式

Stage2 不接收用户级 `evidence_registry_path`。正式入口只消费 Stage1 manifest 和 Stage1 classifier report。Stage2 运行中如果产生了新的 TVM/benchmark 证据，应写成 `stage2_evidence_delta_v1`，再由 Stage1 evidence / calibrated predictor / model classifier 链路吸收。

## 最小结构

```json
{
  "schema": "stage2_evidence_delta_v1",
  "model": "pyramid_lidar",
  "records": [
    {
      "backend": "h800_tvm",
      "hardware": "H800 Hopper",
      "scope": "rsu_dense_core",
      "evidence_kind": "measured",
      "provenance": "tvm_metaschedule_run_001",
      "candidate_config": {
        "software_candidate": "bev_encoder.s2",
        "width": 128,
        "quant": "fp16"
      },
      "metric": {
        "latency_us": 5506.57
      },
      "promotable_to_classifier": true
    }
  ],
  "promotable_record_count": 1,
  "no_overpromotion": true
}
```

## 字段规则

- `backend`: 新增实测默认只允许 `h800_tvm`。
- `hardware`: 实际测量硬件名称。
- `scope`: 证据范围，例如 `rsu_dense_core`。dense-core 证据不能写成 full-model 结论。
- `evidence_kind`: 取 `measured`、`demo`、`proxy`、`historical` 或 `estimated`。
- `provenance`: 生成证据的脚本、run id 或 fixture 名称。
- `candidate_config`: 被测软件/硬件配置。
- `metric`: 延迟、build 状态、AP 或其他指标。

## 升级规则

- 只有 `backend=h800_tvm` 且 `evidence_kind=measured` 的记录可进入 classifier refresh。
- `demo`、`proxy`、`historical`、`estimated` 都不能升级成 pass 结论。
- TRT 只能作为 `historical` 或独立 backend scope，不能写成 Stage1/Stage2 新实测后端。
- `scope=rsu_dense_core` 或其他 dense-core 范围不能支持 full-model speedup claim。

## 验证命令

```bash
PYTHONPATH=. python scripts/stage2_update_evidence.py \
  --delta results/stage2/pyramid/stage2_evidence_delta_v1.json \
  --out-dir results/stage1_model_predict/stage2_evidence_delta
```

该命令只验证和归档 delta，不会自动把 demo/proxy/historical 证据改写为 measured，也不会自动刷新 classifier 结论。
