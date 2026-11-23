import ctypes, sys, time
from collections import deque
import tensorrt as trt

PLUGIN = "./DL4AGX_min/uniad-trt/inference_app/enqueueV2/build/libuniad_plugin.so"
ONNX   = "./onnx/uniad_tiny_imgx0.25_cp.noaten_simp.onnx"
ENGINE = "uniad_tiny_fp16_fixnan.engine"
N = 1150
CONSTRAINTS = sys.argv[1] if len(sys.argv) > 1 else "obey"

ctypes.CDLL(PLUGIN, mode=ctypes.RTLD_GLOBAL)
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

t0 = time.time()
with open(ONNX, "rb") as f:
    data = f.read()
print("parsing %d bytes ..." % len(data), flush=True)
if not parser.parse(data):
    for i in range(parser.num_errors):
        print("PARSE ERR:", parser.get_error(i), flush=True)
    raise SystemExit("parse failed")
print("parsed in %.1fs, layers=%d" % (time.time() - t0, network.num_layers), flush=True)

# ---- locate bev_embed output tensor ----
bev = None
for i in range(network.num_outputs):
    if network.get_output(i).name == "bev_embed":
        bev = network.get_output(i)
assert bev is not None, "bev_embed not a network output"

# ---- producer map: tensor name -> layer index ----
prodmap = {}
for i in range(network.num_layers):
    L = network.get_layer(i)
    for k in range(L.num_outputs):
        t = L.get_output(k)
        if t is not None:
            prodmap[t.name] = i

# ---- backward BFS over TRT layers from bev_embed (by index, stable) ----
keep_ids = set()
dq = deque()
bi = prodmap.get(bev.name)
if bi is not None:
    dq.append(bi)
while dq:
    idx = dq.popleft()
    if idx in keep_ids:
        continue
    keep_ids.add(idx)
    L = network.get_layer(idx)
    for j in range(L.num_inputs):
        t = L.get_input(j)
        if t is not None:
            pi = prodmap.get(t.name)
            if pi is not None:
                dq.append(pi)
print("keep (ancestors of bev_embed) layers = %d" % len(keep_ids), flush=True)

# ---- optimization profile: bind dynamic (-1) dims to N ----
profile = builder.create_optimization_profile()
for i in range(network.num_inputs):
    t = network.get_input(i)
    shape = list(t.shape)
    if -1 in shape:
        s = tuple(N if d == -1 else d for d in shape)
        profile.set_shape(t.name, s, s, s)
        print("profile %s -> %s" % (t.name, s), flush=True)

config = builder.create_builder_config()
config.add_optimization_profile(profile)
config.set_flag(trt.BuilderFlag.FP16)
if CONSTRAINTS == "obey":
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
elif CONSTRAINTS == "prefer":
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
try:
    WS = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WS << 30)
    print("workspace cap = %d GiB" % WS, flush=True)
except Exception as e:
    print("workspace set warn:", e, flush=True)

# ---- force fp32 on every non-ancestor float-output layer ----
forced = 0
skipped_int = 0
skipped_plugin = 0
PLUGIN_TYPES = (trt.LayerType.PLUGIN, trt.LayerType.PLUGIN_V2)
for i in range(network.num_layers):
    L = network.get_layer(i)
    if i in keep_ids:
        continue
    if L.type in PLUGIN_TYPES:
        skipped_plugin += 1
        continue
    is_float = any(L.get_output_type(k) in (trt.float32, trt.float16)
                   for k in range(L.num_outputs))
    if not is_float:
        skipped_int += 1
        continue
    try:
        L.precision = trt.float32
        for k in range(L.num_outputs):
            L.set_output_type(k, trt.float32)
        forced += 1
    except Exception as e:
        skipped_int += 1
print("forced fp32 layers=%d  skipped non-float=%d  skipped plugin=%d  constraints=%s"
      % (forced, skipped_int, skipped_plugin, CONSTRAINTS), flush=True)

t1 = time.time()
print("building engine ...", flush=True)
plan = builder.build_serialized_network(network, config)
if plan is None:
    raise SystemExit("BUILD FAILED (returned None)")
with open(ENGINE, "wb") as f:
    f.write(plan)
print("BUILT ok in %.1fs -> %s (%d bytes)" % (time.time() - t1, ENGINE, len(plan)), flush=True)
