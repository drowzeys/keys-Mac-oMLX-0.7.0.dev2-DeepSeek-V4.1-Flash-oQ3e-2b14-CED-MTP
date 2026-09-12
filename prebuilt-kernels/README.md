# Prebuilt oMLX custom Metal kernels

A `git clone` of oMLX ships **no compiled kernels**, so `fast.is_native_available()` is `False` and
every custom-kernel fast path silently falls back to generic MLX ops — no error, just ~50% of the
prefill throughput. Building them needs Xcode's Metal toolchain, a nanobind pinned to MLX's ABI, and
cmake. These are that build's output, so you can skip it.

Worth **+52% prefill / −34% TTFT** on a 763B MoE (227 → 346 tok/s before CED prefill).

## ⚠️ ABI — these only work on a matching stack

| | |
|---|---|
| MLX | **0.32.2** (exact — kernels are ABI-coupled) |
| Python | **3.11** (`cp311` tags) |
| nanobind (build-time) | 2.15.0 |
| Platform | macOS 26.6.2, arm64 (Apple Silicon) |

A mismatched MLX or Python **will not** warn you — you get an import failure, or worse, a silent
fallback. Always assert after installing.

## Install

```bash
./install.sh /path/to/omlx        # the oMLX checkout you serve from
```

## Verify — do this every time, and before any benchmark

```bash
python -c "
from omlx.custom_kernels.glm_moe_dsa import fast as f
assert f.is_native_available(), 'kernels NOT active - you are on the slow fallback path'
assert f.has_symbol('deepseek_affine_gather_qmm_blocks')
print('native kernels active')"
```

If that assert fails, rebuild from source instead — see `../scripts/setup_all_kernels.py`.
