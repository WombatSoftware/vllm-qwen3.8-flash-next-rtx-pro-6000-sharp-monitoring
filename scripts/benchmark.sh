#!/usr/bin/env bash
# Reproduce the sweep in docs/benchmarks.md against a running engine.
#
#   pip install llama-benchy
#   ./scripts/benchmark.sh [output-id]
#
# Takes 20-25 minutes. The engine must already be healthy -- a run against a
# restarting engine produces meaningless numbers. llama-benchy buffers stdout
# until exit, so poll `pgrep -f llama-benchy` for liveness, not log growth.
#
# The sweep runs under scripts/bench-guard.sh, which waits for the engine to go
# idle and aborts (exit 42) if any other client touches it mid-run. Keep the
# $OUT/$ID.guard verdict next to the results; a sweep without `VERDICT: PASS`
# is not evidence. If the engine also serves agents, run this only while none
# of them are active.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

ID="${1:-local}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix}"
SERVED="${SERVED:-qwen-3-8-flash-next}"
OUT="${OUT:-./bench-results}"
BENCHY="${BENCHY:-llama-benchy}"
CONTAINER="${CONTAINER:-qwen-3-8-flash-next}"
# Host requests to a published port reach the engine NAT'd to the compose
# network's gateway, so that is the address the bench itself logs as.
GATEWAY="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.Gateway}}{{end}}' "$CONTAINER" 2>/dev/null)"
ALLOW_IPS="${ALLOW_IPS:-127.0.0.1${GATEWAY:+,$GATEWAY}}"

if ! curl -sf -o /dev/null "${BASE_URL%/v1}/health"; then
  echo "engine is not healthy at ${BASE_URL%/v1}/health -- start it first" >&2
  exit 1
fi
mkdir -p "$OUT"

echo "running the full sweep -> $OUT/$ID.md (20-25 min)"
CONTAINER="$CONTAINER" ALLOW_IPS="$ALLOW_IPS" EXPECTED_MAX=4 \
METRICS_URL="${BASE_URL%/v1}/metrics" REPORT="$OUT/$ID.guard" \
./scripts/bench-guard.sh -- "$BENCHY" \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --served-model-name "$SERVED" \
  --pp 2048 --tg 128 --depth 0 8192 32768 131072 --concurrency 1 4 --runs 3 \
  --latency-mode generation --enable-prefix-caching \
  --format md --save-result "$OUT/$ID.md" > "$OUT/$ID.log" 2>&1
rc=$?
echo "$rc" > "$OUT/$ID.done"
echo "exit $rc -- results $OUT/$ID.md, log $OUT/$ID.log, guard $OUT/$ID.guard"
exit $rc
