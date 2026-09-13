# Memory budget and context

**Corrected.** An earlier revision of this file claimed a ~40K context ceiling. That was wrong about
the cause, and the fix changed the answer entirely: **404,805 tokens verified at 805 tok/s.**

## KV storage is not the constraint

The architecture is built for 1M (YaRN factor 16 over a 65,536 base), and its KV is tiny because only
**4 layers** (`kv_source_layer_ids = [2, 8, 14, 20]`) produce shared compressed KV; every layer's
local attention is a fixed 128-token sliding window that never grows:

| context | global KV (~2.0 KB/token, shared) |
|---:|---:|
| 65,536 | 0.13 GB |
| 262,144 | 0.54 GB |
| 1,048,576 | 2.15 GB |

## The real constraint is the prefill working set

Prefill costs **~505 KB/token** of *transient* activation memory for the chunk in flight. When the
model leaves too little headroom, oMLX does not fail — it shrinks the chunk:

```
Prefill throttled: chunk 2048 -> 32 (usage 218.18GB vs sizing target 213.61GB, per_token=505.0KB)
```

At a 32-token floor a 41K prompt needs ~1,300 chunks. That is what looked like a context ceiling.
**It was a throughput collapse, not a capacity limit.**

Measured, same machine, same prompt:

| prompt | 14-layer (197.8 GiB free → throttled) | **27-layer (46 GB headroom)** |
|---:|---:|---:|
| 25,949 tok | 217.6 s @ 119 tok/s | **45.0 s @ 576 tok/s** |
| 80,996 tok | — | 97.4 s @ 831 tok/s |
| 212,256 tok | — | 252.6 s @ 840 tok/s |
| **404,805 tok** | — | **502.7 s @ 805 tok/s** |

Prefill gets *faster* at longer context (better chunk amortization), then holds ~800–840 tok/s.

## Budget on a 256 GB Mac Studio

| | |
|---|---|
| Machine RAM | 238.4 GiB (256 GB) |
| Metal wired cap | 248.0 GiB (`iogpu.wired_limit_mb=253952`) |
| Enforcer ceiling (tier=safe) | ~243.8 GB |
| This model, resident | **197.19 GB** |
| **Headroom** | **~46 GB** — enough to keep a full 2048-token chunk |

Practical guidance: **500K+ is reachable; budget ~8.4 min of prefill for 400K.** Set your client's
declared context to what you will actually use — an unbounded window means history grows until the
chunk throttles.

## ⚠️ Two traps that masquerade as OOM

1. **A stale server holds the RAM.** oMLX renames itself via `setproctitle` to **`omlx-server`**, so
   `pkill -f "omlx.cli"` misses it and it survives, holding ~200 GB. The next launch then reports
   `dynamic memory ceiling (25.24GB) ... only 25.09GB is reclaimable` and returns **HTTP 507**.
   Always match `"omlx-server|omlx.cli"`, and `kill -9` (it ignores SIGTERM).
2. **`du -sh` lies across hardlinked checkpoints.** Check `st_nlink`. And check the rewrite payload
   *before* building: a from-scratch 27-layer build needs **179.6 GiB**, but building incrementally
   from the 14-layer variant needs only **91.4 GiB**.
