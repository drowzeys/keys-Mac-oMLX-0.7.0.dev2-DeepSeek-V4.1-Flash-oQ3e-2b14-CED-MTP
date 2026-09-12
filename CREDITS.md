# Credits

**Almost everything here is other people's work.** This repo contributes a layer-selection
strategy, a 30-second requant script, one optional runtime patch, and a set of measurements.
The model, the quantization that made it tractable, the runtime, the kernels, and the framework
underneath all come from the people below. Thank you.

---

## DeepSeek-AI — the model

[**DeepSeek-V4.1-Flash**](https://huggingface.co/deepseek-ai) — the 763B MoE this is all built on:
the CSA2 sparse-attention design, the Engram n-gram tables, the DSpark MTP drafter, and the
`sqrtsoftplus` routing. Everything downstream is a packaging exercise on top of their research.

## Jundot — oMLX runtime *and* the quantized checkpoints

Two distinct debts, either of which alone would have made this impossible.

**[oMLX](https://github.com/jundot/omlx)** (Apache-2.0) — "LLM inference server, optimized for your
Mac." The entire serving stack: the `deepseek_v41` model implementation, Engram SSD offload, the
custom Metal kernels (`glm_moe_dsa`, `decode_fast`, `qwen35_prefill`, `bonsai`, `minimax_m3`), the
paged KV cache, the oQ mixed-precision quantizer, DSpark MTP integration, and the memory enforcer
whose refusal message is what finally located our prefill ceiling.

Specific work this build depends on:

| PR | What it gave us |
|---|---|
| [#3574](https://github.com/jundot/omlx/pull/3574) | DeepSeek V4.1 Flash + DSpark MTP + Engram SSD offload — the foundation |
| [#3607](https://github.com/jundot/omlx/pull/3607) | CED prefill skip with SWA bounded replay — **+45% prefill, −31% TTFT** |
| [#2595](https://github.com/jundot/omlx/pull/2595) | MoE expert offload — streams non-resident experts from the checkpoint |
| [#3571](https://github.com/jundot/omlx/pull/3571) | CSA2 + Engram support (closed, but informed the design) |

**[`Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp`](https://huggingface.co/Jundot/DeepSeek-V4.1-Flash-oQ3e-mtp)**
— the calibrated mixed-precision checkpoint this one is derived from. Their oQ pipeline
(sensitivity measurement, imatrix calibration, the official mixed-bit allocator) did the hard
quantization work. **231.1 GiB of our 309 GB is byte-identical to theirs** — we changed 14 layers and
kept the rest exactly as they built it. Their [`oQ4e`](https://huggingface.co/Jundot/DeepSeek-V4.1-Flash-oQ4e-mtp)
checkpoint and its model card also supplied the residency analysis that framed the whole problem.

## Apple / ml-explore — the framework and the silicon

* **[MLX](https://github.com/ml-explore/mlx)** (MIT) — the array framework and Metal backend.
  Pinned at 0.32.2 here; the custom kernels are ABI-coupled to it.
* **[mlx-lm](https://github.com/ml-explore/mlx-lm)** (MIT) — generation, `BatchGenerator`, KV cache
  primitives, quantization ops.
* **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)** — the multimodal path.
* Apple's **Metal** toolchain and the M3 Ultra's unified memory, without which a 763B model on one
  desktop machine is not a conversation anyone gets to have.

## Others in the dependency chain

* **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)** — Prince Canuma
* **[nanobind](https://github.com/wjakob/nanobind)** — Wenzel Jakob; the C++/Python binding layer the
  Metal kernels are built against (ABI-pinned to 2.15.0)
* **[Hugging Face](https://huggingface.co)** — `transformers`, `tokenizers`, `huggingface_hub`, and
  the Hub itself. Their xet deduplication meant publishing a 309 GB checkpoint cost ~9.5 GB of new
  data instead of re-uploading 231 GiB of Jundot's bytes.
* `numpy`, `sentencepiece`, `tiktoken`, `mistral-common`, `regex`, `jinja2`, `rich`, `psutil` and the
  rest of the oMLX dependency set.

---

## What is actually ours

For honesty, the short list:

* `scripts/build_2b.py` — shard-local 3b→2b requant that hardlinks untouched shards
* `scripts/sensitivity_3to2.py` — per-layer damage measurement, and the finding that it is **useless**
  as a selector (0.0037 spread) so layer choice must be structural
* `omlx-patch/expert_cache.py` — SSD-paged routed experts, MTP-compatible (upstream's refuses to run
  with DSpark MTP), verified bit-identical
* `scripts/bench_v41.py` — a benchmark harness that counts reasoning-model tokens correctly
* `SERVING.md` — the wired-limit and `omlx.cli` findings
* The measurements, and the negative results: dual-ANE is a 1.7% ceiling here, Engram is 0.4% of
  runtime, decode is not batched, and paging throughput depends only on the RAM deficit.

Bugs and mistakes in this repo are ours, not theirs. If something here is wrong, it is not a
reflection on the upstream projects.
