#!/bin/bash
# Standing launch for DeepSeek-V4.1-Flash-2b14 on cityhunter.
#
# MUST be omlx.cli (not `python -m omlx.server`): only the CLI sets
# scheduler_config.paged_ssd_cache_dir, which is what enables prefix reuse.
# Without it hermes re-prefills its 15.7K-token system prompt every call (~99s).
#
# --initial-cache-blocks 8 : the default 256 reserves too much up front and SIGKILLs the load.
# --memory-guard safe      : at wired_limit 253952 it no longer rejects, and it converts
#                            silent SIGKILLs into explicit numeric refusals.
# Requires: sudo sysctl iogpu.wired_limit_mb=253952  (prefill cap is 90% of this)
set -u
cd "$HOME/dsv41-work" || exit 1
mkdir -p kvcache logs
if pgrep -f "omlx.cli|omlx-server" >/dev/null; then
  echo "already running (pid $(pgrep -f 'omlx.cli|omlx-server' | head -1))"; exit 0
fi
LIMIT=$(sysctl -n iogpu.wired_limit_mb)
[ "$LIMIT" -ge 253952 ] || echo "WARNING: iogpu.wired_limit_mb=$LIMIT (<253952); prefill may be refused"
nohup env OMLX_DSV41_EXPERT_CEILING_GB=0 PYTHONPATH="$HOME/omlx-dev2" \
  "$HOME/v41venv/bin/python" -m omlx.cli serve \
    --model-dir "$HOME/.omlx/models" --host 0.0.0.0 --port 11601 \
    --paged-ssd-cache-dir "$HOME/dsv41-work/kvcache" --paged-ssd-cache-max-size 16GB \
    --initial-cache-blocks 8 --memory-guard safe \
  >>logs/serve.log 2>&1 &
echo "launched pid $! -> http://192.168.1.243:11601/v1 (model dsv41-flash-2b14)"
