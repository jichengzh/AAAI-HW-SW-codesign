#!/bin/bash
# 回滚 N=901 实验: 删除 _n901 引擎与日志, 校验 1150 引擎未被改动。
set -u
cd "$(dirname "$0")/.." || exit 1
rm -f uniad_tiny_*_n901.engine bench_n901_*.log build_fp16.log build_fp32.log build_int8_qdq_cnn_nofp16.log
echo "removed n901 engines/logs."
echo "=== 校验 1150 引擎完整性 ==="
md5sum -c tools_acc/baseline_1150.md5
