import tensorrt as trt
from pathlib import Path
LOG=trt.Logger(trt.Logger.ERROR)
class C(trt.IInt8MinMaxCalibrator):
    def __init__(s,c): trt.IInt8MinMaxCalibrator.__init__(s); s.c=Path(c).read_bytes()
    def get_batch_size(s): return 1
    def get_batch(s,n): return None
    def read_calibration_cache(s): return s.c
    def write_calibration_cache(s,c): return None
JOBS=[("base","base.onnx","base_int8_calib_trt85.cache"),
      ("p50","pruned50.onnx","pruned50_int8_calib_trt85.cache"),
      ("p75","pruned75.onnx","pruned75_int8_calib_trt85.cache")]
for tag,onnx,cache in JOBS:
    out=f"{tag}_int8_orin.engine"
    b=trt.Builder(LOG); net=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    p=trt.OnnxParser(net,LOG); p.parse(Path(onnx).read_bytes())
    cfg=b.create_builder_config(); cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2<<30)
    cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator=C(cache)
    ser=b.build_serialized_network(net,cfg)
    if ser is None: print(f"[FAIL] {tag}"); continue
    Path(out).write_bytes(bytes(ser)); print(f"[OK] {out} {Path(out).stat().st_size/1e6:.2f}MB")
print("RETRY_DONE")
