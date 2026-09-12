# Serving notes

Everything here was measured on the target box (Mac15,14 / M3 Ultra / 256 GB / macOS 26.6.2).
Two of these cost hours to find; neither is documented upstream.

## 1. Raise the kernel wired limit — or nothing works

oMLX sets its **prefill safety cap at 90% of `iogpu.wired_limit_mb`**. The stock limit on a 256 GB
Mac is `249036` (243.2 GiB), which puts the cap at **218.4 GB** — *below* this 217.95 GB model.
The result is 0.5 GB of working headroom, and the server either refuses every prefill:

```
Prefill capacity rejected: predicted peak would exceed prefill safety cap 218.4GB
(90% of metal_cap ceiling 242.7GB) ... usage 217.95GB vs sizing target 218.43GB
```

(which surfaces to an OpenAI client as the very misleading **"Context length exceeded (24 tokens)"**)
or, with the guard off, dies with a silent **SIGKILL / exit 137** partway through loading.

```bash
sudo sysctl iogpu.wired_limit_mb=253952      # 248.0 GiB -> cap ~223 GiB, ~20 GiB headroom
```

Does **not** persist across reboot. `scripts/serve_dsv41.sh` warns when it is unset.

## 2. Launch through `omlx.cli serve`, never `python -m omlx.server`

Only the CLI sets `scheduler_config.paged_ssd_cache_dir`, which is what enables **prefix reuse**.
The bare server module silently logs

```
oMLX cache disabled (mlx-lm BatchGenerator manages KV internally)
```

and there is **no env var that fixes it** — `OMLX_PAGED_SSD_CACHE_DIR` populates a *different* object
(`config.paged_ssd_cache`) that the scheduler never reads.

This matters enormously for agent workloads, which resend a large static system prompt every call.
Measured with an agent harness sending an identical **15,688-token** prompt:

| | `python -m omlx.server` | `omlx.cli serve` |
|---|---:|---:|
| server-side, cold | 98.9 s | 56.3 s |
| server-side, **warm** | 98.9 s (no reuse) | **6.6 s** |
| client wall, steady state | ~100 s | **7.4–8.7 s** |

A 2-token reply was taking ~99 s of pure prefill. Nothing to do with decode speed.

⚠️ **The SSD cache dir stays 0 B / 0 files.** Blocks remain hot in RAM and nothing needs evicting.
Do not read an empty `kvcache/` as "caching is broken" — read the prompt-timing delta instead.

## 3. Flags that matter

* `--initial-cache-blocks 8` — the **default 256** reserves too much up front and SIGKILLs the load.
* `--memory-guard safe` — at the raised wired limit it no longer rejects, and it converts silent
  SIGKILLs into explicit, numeric refusals. That refusal text is what located the cap in the first
  place. Keep it on.
* `--paged-ssd-cache-max-size 16GB` — plenty; see the 0 B note above.

## 4. Model settings (`~/.omlx/model_settings.json`)

See `model_settings.json`. `deepseek_v41_ced_prefill_enabled` is worth **+45% prefill / −31% TTFT**;
it requires an even layer count, `mid` in both kv- and index-source layers, and
`compress_ratios[mid:] == 1`. Confirm with `CED prefill enabled: decoder tail 128` in the log.

## 5. Operational gotchas

* **First request after idle costs ~2×** (6.8 s vs 2.9 s for 64 tokens) — resident pages migrate to
  the compressed pool while idle and fault back. Send a throwaway prompt first; always discard rep 0.
* **`pkill -f omlx.server` misses the running server** — it re-execs with argv[0] `omlx-server`, and a
  stale instance then holds the port so the next launch dies with `EXIT_CODE=3 ... address already in
  use`. Match `"omlx.cli|omlx-server"`.
* **Headroom is not optional.** A 10-layer variant (224.08 GB resident) swapped under sustained 8-way
  load — free memory hit 9%, swap climbed 5 MB → 1.5 GB, throughput collapsed (prose c8 20.1 → 11.0).
  The 14-layer build holds swap flat.
* **Model residency is right at the edge even so.** At 217.77 GB it loaded reliably only with the
  raised wired limit; before that it managed exactly one load on a freshly rebooted machine (swap 0)
  and failed every attempt once swap reached ~1.2 GB.
* **Verify the cache actually initialized**, not that the process survived. Assert the
  `paged SSD cache enabled: ...` log line.
