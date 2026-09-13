# keys-Mac-oMLX-0.7.0.dev2-DeepSeek-V4.1-Flash-oQ3e-2b27-CED-MTP

Running **DeepSeek-V4.1-Flash (763B)** on a **single 256 GB Mac Studio M3 Ultra** — a checkpoint
that does not fit the machine as published — plus the tooling and measurements that got it there.

| | |
|---|---|
| **Weights** | [`drowzeys/keys-Mac-oMLX-0.7.0.dev2-DeepSeek-V4.1-Flash-oQ3e-2b27-CED-MTP`](https://huggingface.co/drowzeys/keys-Mac-oMLX-0.7.0.dev2-DeepSeek-V4.1-Flash-oQ3e-2b27-CED-MTP) — 309 GB, ready to serve |
| Base checkpoint | [`Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp`](https://huggingface.co/Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp) (330.90 GiB) |
| Runtime | [oMLX](https://github.com/jundot/omlx) **0.7.0.dev2**, commit `395ec2fd` |
| MLX | **0.32.2** (the kernels are ABI-coupled to this exact version) |
| Host | Mac15,14 / M3 Ultra / 256 GB unified / macOS 26.6.2 |
| Resident weights | **197.19 GB** (Engram on NVMe, zero expert paging) |
| Speculation | DSpark MTP, built into the checkpoint, `mtp_num_draft_tokens=7` |
| Serving | see **[SERVING.md](SERVING.md)** — two non-obvious requirements |
| **Verified context** | **404,805 tokens** @ 805 tok/s prefill — see [LIMITS.md](LIMITS.md) |

## Why a rebuild was needed

`oQ3e` is 330.90 GiB total. Even with Engram SSD-offload — the model card's own mitigation, which
moves 91.86 GiB of n-gram tables to disk — **239.04 GiB of weights stay resident**, against
238.42 GiB of usable RAM. It does not fit, by about a gigabyte, before KV and activations.

Two levers were measured. Only one of them works:

* **SSD-paging the routed experts** — implemented and correct (see `omlx-patch/`), but it caps the
  whole system at **~11 tok/s**. Miss traffic works out to `N x 88.6 MiB x deficit/(N x 5.54 GiB)`
  — **`N` cancels**. Paging throughput depends *only* on the RAM deficit, never on how many layers
  you page or how large you make the cache, because caching more forces evicting more.
* **Shedding bytes** — requantize a subset of MoE layers `affine 3-bit -> 2-bit @ group_size 64`.
  This is what ships here.

## The build

**27 of 40** MoE layers, routed experts only (every layer that is not structurally load-bearing):

```
3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19,
21, 22, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35
```

Sheds **42.7 GiB** across 81 modules. Built incrementally from the 14-layer variant in ~40 s. Only the 20 shards holding those layers are
rewritten; the other 45 are hardlinked.

### Layer choice is structural, not numerical

Per-layer weight-cosine damage from `3b -> 2b` is **essentially flat** — the entire 40-layer spread
is **0.0037** (0.9287–0.9323). That metric cannot discriminate, and trusting it caused a real
regression: the first attempt picked layers 25–34, which includes **28 and 32 — members of
`index_source_layer_ids`**. Those layers produce the index keys for sparse attention, so 2-bit there
corrupts *which tokens attention selects*. MTP acceptance collapsed **84% -> 58.5%**. Excluding them
restored it.

**Never quantize below 3-bit:** `kv_source_layer_ids`, `index_source_layer_ids`,
`candidate_source_layer_id`, `engram_layer_ids`, `dspark_target_layer_ids`, or layers `0` / `n-1`.
The DSpark drafter is target-attached — change the target and you desynchronize a drafter you
cannot retrain.

**MTP weights are never touched.** A degraded drafter is what sank the 86 GB 2.4-bit V4 checkpoint
on code despite being 1.8x smaller.

## Results

Single stream, 256 max_tokens, temp 0:

| | paged 3b | 14-layer 2b | **this build (27-layer)** |
|---|---:|---:|---:|
| prose | 15.45 | 25.69 | **25.08** tok/s |
| code | 10.18 | 29.08 | **29.85** tok/s |
| TTFT (short) | 3.53 s | 0.74 s | **0.78 s** |
| resident | — | 217.77 GB | **197.19 GB** |

Concurrency (aggregate tok/s):

| task | c=1 | c=2 | c=4 | c=8 |
|---|---:|---:|---:|---:|
| prose agg | 23.38 | 19.64 | 19.06 | 19.21 |
| code agg | 27.36 | 19.64 | 19.03 | 19.20 |
| per-stream | 25.08 / 29.85 | 10.27 | 4.90 | 2.49 |
| TTFT | 0.78–0.81 s | 1.14 s | 1.46 s | 3.89 s |

All numbers at the shipped `mtp_num_draft_tokens: 3`. MTP acceptance **75.4% code / 59.1% prose**.

**Long context — the reason for 27 layers.** Freeing 20.6 GiB stops oMLX throttling the prefill
chunk, which is what actually gated long prompts:

| prompt | 14-layer (throttled) | **27-layer** |
|---:|---:|---:|
| 25,949 tok | 217.6 s @ 119 tok/s | **45.0 s @ 576 tok/s** |
| 80,996 tok | — | **97.4 s @ 831 tok/s** |
| 212,256 tok | — | **252.6 s @ 840 tok/s** |
| **404,805 tok** | — | **502.7 s @ 805 tok/s** |

### MTP draft depth: use `k=3`, not the default

More 2-bit layers desynchronize the target from the (untouched) DSpark drafter, so acceptance falls.
The fix is **shorter drafts** — with a weak drafter, a long draft just wastes verification. Measured
on prose at temp 0.3:

| `mtp_num_draft_tokens` | tok/s | tok/cycle | acceptance |
|---:|---:|---:|---:|
| **3** | **22.69** | **2.04** | **63.8%** |
| 5 | 20.74 | 1.79 | 54.5% |
| 7 | 19.77 | 1.76 | 52.2% |

End to end that is prose **23.35 → 25.06 tok/s (+7.3%)**, acceptance 55.2 → 59.1%, with code
**unaffected** (29.64 → 29.78). Unusually for spec-decode there is no per-task tradeoff here, so
`k=3` is the shipped default.

⚠️ Temperature also moves acceptance — **up**, counterintuitively (54.3% at temp 0 → 63.8% at
temp 1.0), because rejection sampling accepts more when the target distribution is flatter. But
throughput barely changes, so it is not a useful lever.

**Aggregate saturates ~19.5 regardless of batch, and c=8 is no faster than c=1.** That is not
contention and not I/O — Engram profiling put the n-gram path at 0.4% of runtime with zero lock wait
and a 100% prefetch hit, and the flat curve reproduces with MTP disabled. **Decode is simply not
batched**: each forward serves one request, so aggregate equals the single-stream rate.

Attention/MoE split is 51%/49% — no single hotspot. Decode runs ~3x off its bandwidth bound and
prefill ~3x off its compute bound: a uniform ~30% efficiency, the signature of many small unfused
ops at batch=1.

**For agent workloads the decode numbers are not the story.** A harness resending a static
15,688-token system prompt spent ~99 s per call on prefill alone until prefix reuse was enabled —
then 6.6 s warm. See [SERVING.md](SERVING.md); it is the single highest-leverage thing in this repo.

Where the prefill/TTFT gains came from:

| | prefill (1893 tok) | TTFT |
|---|---:|---:|
| no custom kernels | 227 tok/s | 8.34 s |
| all 5 Metal kernels built | 346 tok/s | 5.48 s |
| + oMLX 0.7.0.dev2 CED prefill | **551 tok/s** | **3.43 s** |

## Reproduce

```bash
# 0. PREREQUISITE — raise the kernel wired limit, or the server refuses every prefill.
#    oMLX caps prefill at 90% of this; the stock 249036 puts the cap BELOW the model.
sudo sysctl iogpu.wired_limit_mb=253952          # does not persist across reboot

# 1. runtime
git clone https://github.com/jundot/omlx && cd omlx && git checkout 395ec2fd
python -m venv .venv && .venv/bin/pip install "mlx==0.32.2" "nanobind==2.15.0" "cmake>=3.27"

# 2. custom Metal kernels — a clone ships NONE, and everything silently falls back without them.
#    Prebuilt (MLX 0.32.2 / cp311 / macOS 26 arm64) — skips the Xcode+nanobind+cmake build:
./prebuilt-kernels/install.sh .
#    ...or build from source if your ABI differs:
# PATH=$PWD/.venv/bin:$PATH DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
# OMLX_WITH_CUSTOM_KERNEL=1 .venv/bin/python setup_all_kernels.py build_ext --inplace
.venv/bin/python -c "from omlx.custom_kernels.glm_moe_dsa import fast as f; assert f.is_native_available()"

# 3. requant (~30 s; needs the base checkpoint on disk)
python scripts/build_2b.py /path/to/oQ3e /path/to/out 7,15,19,21,22,23,25,26,27,29,30,31,33,34

# 4. serve — MUST be omlx.cli, not `python -m omlx.server` (see SERVING.md)
./scripts/serve_dsv41.sh
```

⚠️ `build_2b.py` rewrites every shard holding a target layer and hardlinks the rest, so disk cost
scales with the layer count: 14 layers needs ~61 GiB of free space, 20 layers ~99 GiB. Check first —
overrunning it fills the volume mid-write.

Building the kernels is worth **+52% prefill / −34% TTFT**, and `deepseek_v41_ced_prefill_enabled`
a further **+45% / −31%**. CED requires an even layer count, `mid` in both kv- and index-source
layers, and `compress_ratios[mid:] == 1`.

## `prebuilt-kernels/`

All five oMLX custom Metal kernel extensions, already compiled — 14 MB, drop-in via
`install.sh`. Saves the Xcode/nanobind/cmake build and is worth **+52% prefill / −34% TTFT**.
ABI-locked to MLX 0.32.2 / Python 3.11 / macOS 26 arm64; a mismatch fails silently, so
**always assert `is_native_available()`**. See [prebuilt-kernels/README.md](prebuilt-kernels/README.md).

## `omlx-patch/expert_cache.py`

SSD-paged routed experts for `deepseek_v41`. Not needed by this build — it exists because measuring
it is what proved shedding bytes was the only viable path. Two properties worth noting against
upstream's `feat(moe): offload non-resident experts` (#2595), which raises
`"MoE expert offload cannot enable DSpark MTP"`:

* **It works with DSpark MTP.**
* It gathers the unique routed experts into a compact table **in sorted order** and remaps indices,
  so `sorted_indices` still holds and `deepseek_affine_gather_qmm_blocks` stays eligible — paging
  does not cost the fast kernel.

Verified **bit-identical** to a resident `QuantizedProjection` (4/4 trials, maxabs `0.000e+00`), and
the v41 suite is **552 passed / 0 failed on both pristine and patched**.

## Gotchas worth keeping

(Serving/ops gotchas live in [SERVING.md](SERVING.md).)

* **Benchmark harness.** V4.1 is a reasoning model: tokens arrive as `delta.reasoning_content`
  before any `delta.content`, the server interleaves `model:"keepalive"` chunks with empty content
  that destroy TTFT if counted, and a chunk carries several tokens. Take exact counts from
  `usage.completion_tokens` via `stream_options.include_usage`. Getting this wrong reported
  **0.03 tok/s** where the server logged 11.1.
* **First request after idle costs ~2x** (6.8 s vs 2.9 s for 64 tokens) — resident pages migrate to
  the compressed pool when idle and fault back. Send a throwaway prompt first.
* **Headroom is not optional.** A 10-layer variant (224.08 GB resident, ~6% headroom) swapped under
  sustained 8-way load: free memory hit 9%, swap climbed 5 MB -> 1.5 GB and throughput collapsed
  (prose c8 20.1 -> 11.0). This 14-layer build holds swap flat at ~1.0 GB through the same run.
* **`du -sh` lies across hardlinked checkpoints.** Check `st_nlink` before deleting a build dir.

## Limits

**404,805 tokens verified at 805 tok/s.** KV storage was never the constraint — only 4 layers produce
shared compressed KV (~2.0 KB/token), so 1M context is ~2.15 GB. The constraint is the **prefill
working set** (~505 KB/token): when the model leaves too little headroom oMLX shrinks the prefill
chunk to a 32-token floor, and long prompts collapse to ~119 tok/s. Freeing 20.6 GiB stops that.

See **[LIMITS.md](LIMITS.md)** for the full budget and two traps that masquerade as OOM.

## Credits

This is built almost entirely on other people's work — **DeepSeek-AI** for the model, **Jundot**
for both the [oMLX](https://github.com/jundot/omlx) runtime *and* the calibrated
[oQ3e checkpoint](https://huggingface.co/Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp) we derive from
(231.1 GiB of our 309 GB is byte-identical to theirs), and **Apple/ml-explore** for MLX and Metal.

**See [CREDITS.md](CREDITS.md) for the full list**, including the specific oMLX PRs this depends on
and an honest accounting of the small part that is actually ours.

## License

Tooling here is Apache-2.0. The base checkpoint and oMLX carry their own licenses.
