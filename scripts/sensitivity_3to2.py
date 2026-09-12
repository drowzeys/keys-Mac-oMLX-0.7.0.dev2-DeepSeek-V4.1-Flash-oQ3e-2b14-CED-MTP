"""Per-layer 3b->2b gs64 damage on oQ3e routed experts, from a subsample of experts.

Reads only `SAMPLE` expert rows per projection through TensorFile(rows=...), so the
whole 40-layer sweep touches a few GiB instead of 221.
"""
import json, os, sys, time
import mlx.core as mx, numpy as np
from omlx.patches.deepseek_v41.storage import TensorFile, decode_array

PATH = sys.argv[1]
SAMPLE = int(sys.argv[2]) if len(sys.argv) > 2 else 24
cfg = json.load(open(os.path.join(PATH, "config.json")))
qm = cfg["omlx_deepseek_v41"]["quantized_modules"]
wmap = json.load(open(os.path.join(PATH, "model.safetensors.index.json")))["weight_map"]

mods = {n: s for n, s in qm.items()
        if n.startswith("language_model.layers.") and ".ffn.experts." in n}
rng = np.random.default_rng(0)
readers, layers = {}, {}
t0 = time.time()
for name in sorted(mods, key=lambda n: (int(n.split(".")[2]), n)):
    spec = mods[name]
    fn = wmap[name + ".weight"]
    r = readers.get(fn) or readers.setdefault(fn, TensorFile(os.path.join(PATH, fn)))
    E = r.header[name + ".weight"]["shape"][0]
    rows = np.sort(rng.choice(E, size=min(SAMPLE, E), replace=False))
    parts = []
    for suffix in (".weight", ".scales", ".biases"):
        key = name + suffix
        parts.append(decode_array(*r.read(key, rows=rows)) if key in wmap else None)
    ref = mx.dequantize(parts[0], parts[1], parts[2], group_size=spec.get("group_size", 64),
                        bits=spec["bits"], mode=spec["mode"]).astype(mx.bfloat16)
    q = mx.quantize(ref, group_size=64, bits=2, mode="affine")
    new = mx.dequantize(*q, group_size=64, bits=2, mode="affine").astype(mx.float32)
    a = ref.astype(mx.float32)
    cos = float(mx.sum(a * new) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(new * new)) + 1e-30))
    layers.setdefault(int(name.split(".")[2]), []).append(cos)
    del ref, new, a, q, parts; mx.clear_cache()

scores = {L: sum(v) / len(v) for L, v in layers.items()}
order = sorted(scores, key=lambda L: -scores[L])   # highest cos = least damaged
print(f"swept {len(mods)} modules / {len(scores)} layers in {time.time()-t0:.0f}s "
      f"(sample={SAMPLE} experts)\n")
print("layer  cos(3b->2b)   rank")
for rank, L in enumerate(order):
    print(f"{L:5d}  {scores[L]:.6f}  {'<-- CHEAPEST' if rank < 10 else ''}")
print("\nleast-damaged 10 layers:", sorted(order[:10]))
print("spread: best %.6f  worst %.6f  delta %.6f" %
      (scores[order[0]], scores[order[-1]], scores[order[0]] - scores[order[-1]]))
json.dump({"scores": scores, "chosen": sorted(order[:10])},
          open(os.path.expanduser("~/dsv41-work/logs/sensitivity.json"), "w"), indent=1)
