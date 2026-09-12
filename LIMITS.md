# Memory budget and context limits

**Read this before wiring an agent at it.** The model fits; the *working set* is what runs out.

## The budget on a 256 GB Mac Studio

| | |
|---|---|
| Machine RAM | 238.4 GiB (256 GB) |
| Metal wired cap | 248.0 GiB (`iogpu.wired_limit_mb=253952`) |
| **Hard refuse above** | **223.2 GiB / 239.7 GB** — 90% of the wired cap |
| **Throttles above** | ~207 GiB / **222.7 GB** — oMLX "sizing target", dynamic |
| This model, resident | **217.76 GB** |
| **Left for working set** | **~5 GB** |

Prefill working set costs **~505 KB/token** (KV is already FP8/FP4-packed; this is measured, not
theoretical). So ~5 GB buys roughly:

| context | working set | verdict |
|---|---:|---|
| 10K tok | 5 GB | comfortable |
| 16K tok | 7.9 GB | fine in practice |
| **~40K tok** | 19.7 GB | **observed ceiling** |
| 64K tok | 30.8 GB | throttles hard |
| 256K tok | 123 GB | impossible |

## What "over the limit" looks like

It does **not** crash. oMLX shrinks the prefill chunk to keep the working set bounded:

```
Prefill throttled: chunk 2048 -> 32 (usage 218.18GB vs sizing target 213.61GB, per_token=505.0KB)
```

At a 32-token floor a 41K prompt needs ~1,300 chunks — one request took **226 s**, and the server
looks hung. It is not hung, it is crawling. Symptom to recognise: requests that were 5–8 s suddenly
take minutes, and swap climbs.

## Why this model and not others on the same Mac

Nothing about the context code changed. The weights just leave nothing behind:

| model | resident | left for working set | usable context |
|---|---:|---:|---|
| Qwen3.8-27B oQ4e | 15.8 GiB | 222.6 GiB | 256K+ |
| DeepSeek-V4-Flash oQ4e | 143.4 GiB | 95.0 GiB | 1M |
| **this build** | **202.7 GiB** | **35.7 GiB** | **~40K** |

## Configuring a client

Set the client's declared context to what the box can actually serve, **not** the model's 1M window.
Leaving it at 1M means the client never compresses history, the prompt grows unbounded, and you hit
the throttle wall mid-session.

Hermes enforces a **64K minimum**, so `context_length: 64000` is the floor that is accepted; real
prompts stay ~16K and sessions work, but history beyond ~40K will degrade.

## To actually serve 64K+

The model must come down to **≤ ~190 GB**. Note that **2-bit requant alone cannot get there**: only
27 of 40 MoE layers are quantizable (13 are structurally protected — see the main README), and all 27
at 2-bit still lands at ~210.8 GB. Options:

1. **SSD-page routed experts** (`omlx-patch/expert_cache.py`) — frees ~5.5 GiB per paged layer, at a
   throughput cost that depends only on the RAM deficit.
2. **A lower-bit base checkpoint** (oQ2e-class) if one becomes available.
3. **Accept ~16–40K context**, which is what this build ships as.
