"""Requantize selected MoE layers' routed experts from affine 3b gs64 -> 2b gs64.

Only shards holding a chosen layer's expert tensors are rewritten; every other file is
hardlinked, so the build costs ~40 GiB of writes rather than a 331 GB copy. bits stays in
(2,3) at gs64, so `deepseek_affine_gather_qmm_blocks` remains eligible.
"""
import json, os, shutil, sys, time
import mlx.core as mx

SRC, DST = sys.argv[1], sys.argv[2]
CHOSEN = set(int(x) for x in sys.argv[3].split(","))
cfg = json.load(open(os.path.join(SRC, "config.json")))
qm = cfg["omlx_deepseek_v41"]["quantized_modules"]
wmap = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]

targets = {n: s for n, s in qm.items()
           if n.startswith("language_model.layers.") and ".ffn.experts." in n
           and int(n.split(".")[2]) in CHOSEN}
shards = sorted({wmap[n + ".weight"] for n in targets})
print(f"{len(targets)} modules across {len(shards)} shards -> 2b gs64", flush=True)
os.makedirs(DST, exist_ok=True)

saved = 0
for f in sorted(os.listdir(SRC)):
    s, d = os.path.join(SRC, f), os.path.join(DST, f)
    if not os.path.isfile(s):
        continue                      # skip dirs (e.g. .cache left by hf upload)
    if os.path.exists(d):
        os.remove(d)
    if f not in shards:
        if f == "config.json":
            continue                      # rewritten below
        os.link(s, d)                     # untouched: hardlink, zero cost
        continue
    t0 = time.time()
    values = mx.load(s)
    out, touched = {}, []
    for name in [n for n in targets if wmap[n + ".weight"] == f]:
        spec = qm[name]
        ref = mx.dequantize(values[name + ".weight"], values[name + ".scales"],
                            values.get(name + ".biases"),
                            group_size=spec.get("group_size", 64),
                            bits=spec["bits"], mode=spec["mode"]).astype(mx.bfloat16)
        mx.eval(ref)
        w, sc, b = mx.quantize(ref, group_size=64, bits=2, mode="affine")
        mx.eval(w, sc, b)
        del ref; mx.clear_cache()
        out[name + ".weight"], out[name + ".scales"], out[name + ".biases"] = w, sc, b
        touched += [name + ".weight", name + ".scales", name + ".biases"]
    for k, v in values.items():
        if k not in touched:
            out[k] = v
    mx.save_safetensors(d, out, metadata={"format": "pt"})
    delta = os.path.getsize(s) - os.path.getsize(d)
    saved += delta
    print(f"  {f}: {os.path.getsize(s)/2**30:.2f} -> {os.path.getsize(d)/2**30:.2f} GiB "
          f"({time.time()-t0:.0f}s)", flush=True)
    del values, out; mx.clear_cache()

for name in targets:
    qm[name] = {"bits": 2, "mode": "affine", "group_size": 64,
                **{k: v for k, v in qm[name].items()
                   if k not in ("bits", "mode", "group_size")}}
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)
print(f"\nshed {saved/2**30:.2f} GiB; config updated for {len(targets)} modules")
