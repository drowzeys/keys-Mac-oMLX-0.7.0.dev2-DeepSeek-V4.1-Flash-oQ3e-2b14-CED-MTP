"""Build all five oMLX custom Metal kernel extensions in one pass.

A git clone of oMLX ships NO compiled `_ext*.so`, so `fast.is_native_available()`
is False and every custom-kernel fast path silently falls back to generic MLX ops.
ALWAYS assert is_native_available() before benchmarking.

  pip install "nanobind==2.15.0" "cmake>=3.27"     # nanobind ABI must match mlx
  PATH=$VENV/bin:$PATH \
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  OMLX_WITH_CUSTOM_KERNEL=1 python setup_all_kernels.py build_ext --inplace
"""
import os, sys
from setuptools import setup

os.environ.setdefault("OMLX_WITH_CUSTOM_KERNEL", "1")
from mlx import extension

ca = os.environ.get("CMAKE_ARGS", "")
os.environ["CMAKE_ARGS"] = (
    f"{ca} -DPython_EXECUTABLE={sys.executable} -DPython3_EXECUTABLE={sys.executable}".strip()
)
NAMES = ["bonsai", "decode_fast", "glm_moe_dsa", "minimax_m3", "qwen35_prefill"]
setup(
    name="omlx-all-kernels", version="0.0.0",
    ext_modules=[
        extension.CMakeExtension(f"omlx.custom_kernels.{n}._ext",
                                 sourcedir=f"omlx/custom_kernels/{n}/csrc")
        for n in NAMES
    ],
    cmdclass={"build_ext": extension.CMakeBuild},
)
