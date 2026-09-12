#!/bin/bash
# Drop prebuilt kernels into an oMLX checkout. Usage: ./install.sh /path/to/omlx
set -eu
DEST="${1:?usage: ./install.sh /path/to/omlx}"
[ -d "$DEST/omlx/custom_kernels" ] || { echo "not an oMLX checkout: $DEST"; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
for n in bonsai decode_fast glm_moe_dsa minimax_m3 qwen35_prefill; do
  cp -v "$HERE/$n"/* "$DEST/omlx/custom_kernels/$n/" 2>/dev/null || echo "  (skipped $n)"
done
echo
echo "Now verify (MUST print 'native kernels active'):"
echo "  PYTHONPATH=$DEST python -c \"from omlx.custom_kernels.glm_moe_dsa import fast as f; assert f.is_native_available(); print('native kernels active')\""
