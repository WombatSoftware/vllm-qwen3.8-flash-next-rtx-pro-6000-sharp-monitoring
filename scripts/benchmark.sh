#!/usr/bin/env bash
# Reproduce the sweep in docs/benchmarks.md against a running engine.
#
#   pip install llama-benchy
#   ./scripts/benchmark.sh [output-id]
#
# Takes 20-25 minutes. The engine must already be healthy -- a run against a
# restarting engine produces meaningless numbers. llama-benchy buffers stdout
# until exit, so poll `pgrep -f llama-benchy` for liveness, not log growth.
set -uo pipefail

ID="${1:-local}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix}"
SERVED="${SERVED:-qwen-3-8-flash-next}"
OUT="${OUT:-./bench-results}"
BENCHY="${BENCHY:-llama-benchy}"

if ! curl -sf -o /dev/null "${BASE_URL%/v1}/health"; then
  echo "engine is not healthy at ${BASE_URL%/v1}/health -- start it first" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "running the full sweep -> $OUT/$ID.md (20-25 min)"
"$BENCHY" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --pp 2048 --tg 128 --depth 0 8192 32768 131072 --concurrency 1 4 --runs 3 \
  --latency-mode generation --enable-prefix-caching \
  --format md --save-result "$OUT/$ID.md" > "$OUT/$ID.log" 2>&1
rc=$?
echo "$rc" > "$OUT/$ID.done"
echo "exit $rc -- results $OUT/$ID.md, log $OUT/$ID.log"
exit $rc
