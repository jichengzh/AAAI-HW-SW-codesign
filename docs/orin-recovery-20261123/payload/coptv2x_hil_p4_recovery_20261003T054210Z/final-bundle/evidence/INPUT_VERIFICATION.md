# P4 输入完整性核验

本轮复用已部署缓存，不重新采集或替换历史输入。

- Host index: `/data/jichengzhi_v2x/ditl/coptv2x_hil_p4_smoke_20261001T074559Z/evidence/host-input-index.json`
- Host index SHA256: `3222c599a0859ede56454f70324a6d600e13e0a87c859a8b76480e9428fc7016`
- Orin cache: `/home/jichengzhi/coptv2x_hil_p4_smoke_20261001T074559Z/input`
- Orin manifest SHA256: `4e59fa61d379a3c9eace433bd8d58e1f15f595bd4cea60cd4351fed226f581fa`
- Exact manifest mapping: 2225 unique entries; tensor SHA256, shape and dtype agree.
- Source storage: 200 entries are in host `spill-input`; the other 2025 are in `capture-full`. The 2225 entries have 2225 distinct tensor SHA256 values; this is storage reuse, not repeated-frame padding.
- Byte verification: all 2225 NPY entries returned `OK` from `sha256sum -c SHA256SUMS`; no non-OK entry.
- Evidence: `orin-input-byte-verification.log`
- Evidence SHA256: `be226615e8933899f2301ee5a5dff8d0b2da16cd7e088093b28310de3cd6baa9`

该证明只说明输入缓存完整且一致。缓存输入不是当前在线车辆重新采集的感知输入，不构成论文级闭环性能比较的有效性证明。
