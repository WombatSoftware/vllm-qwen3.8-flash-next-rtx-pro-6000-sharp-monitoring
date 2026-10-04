#!/usr/bin/env bash
# bench-guard.sh — isolation guard for llama-benchy sweeps (see docs/benchmarks.md, "Benchmark isolation").
#
# Pre-checks the engine is idle, launches the wrapped command, and ABORTS it (SIGINT,
# exit 42) the moment anything violates bench isolation:
#   - a request reaches the engine from a client outside ALLOW_IPS (access-log evidence),
#   - vllm:num_preemptions_total increases (NOTED, not fatal: the bench itself preempts
#     at max-depth c4, where 4xdepth exceeds the KV pool — 4x131072 > 468296 fp8 tokens);
#   - vllm:num_requests_running exceeds EXPECTED_MAX (foreign concurrency; 0 at start).

# Usage:  scripts/bench-guard.sh -- ./venv/bin/llama-benchy <bench args...>
# Env:    CONTAINER (qwen-3-8-flash-next)  METRICS_URL (http://127.0.0.1:8000/metrics)
#         ALLOW_IPS (127.0.0.1 + bridge gateway, comma list). Host-originated requests to
#         published ports are docker-proxy-NAT'd to the bridge gateway IP (172.22.0.1 on
#         this network), so the bench itself logs as the gateway, NOT 127.0.0.1; real LAN
#         clients keep their true IP, so allowing
#         the gateway still detects them. EXPECTED_MAX (required: bench's max concurrency)
# Exit:   wrapped command's exit code; 42 on a recorded violation. VERDICT line appended
#         to REPORT either way; keep it next to the sweep files.
set -uo pipefail

CONTAINER="${CONTAINER:-qwen-3-8-flash-next}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8000/metrics}"
ALLOW_IPS="${ALLOW_IPS:-127.0.0.1,172.22.0.1}"
EXPECTED_MAX="${EXPECTED_MAX:-}"
POLL="${POLL:-5}"
REPORT="${REPORT:-/tmp/bench-guard.report}"

if [[ "${1:-}" != "--" ]]; then
  sed -n '2,17p' "$0"; exit 2
fi
shift
CMD=("$@")

ip_grep=$(echo "$ALLOW_IPS" | tr ',' '\n' | sed 's/[.]/\\./g' | paste -sd'|' -)
rm -f "$REPORT"

metric() {  # sum of all series matching $1
  curl -sf "$METRICS_URL" | awk -v m="$1" '$1 ~ m && $0 !~ /^#/ {s+=$2} END{printf "%.0f", s+0}'
}
fail() {
  printf '%s VIOLATION: %s\n' "$(date -u +%H:%M:%S)" "$*" >> "$REPORT"
  printf '%s VERDICT: FAIL (guard)\n' "$(date -u +%H:%M:%S)" >> "$REPORT"
  if [[ -n "${CHILD:-}" ]]; then
    kill -INT "$CHILD" 2>/dev/null
    # llama-benchy can defer SIGINT past in-flight requests; escalate or it orphans
    # and writes an UNGUARDED result file after the guard already exited 42 (09-24).
    ( sleep 10; kill -TERM "$CHILD" 2>/dev/null
      sleep 5; kill -KILL "$CHILD" 2>/dev/null ) & disown
  fi
  exit 42
}

[[ -z "$EXPECTED_MAX" ]] && fail "set EXPECTED_MAX to the bench's highest concurrency"
# Wait (don't fail) for the engine to drain: the turn that LAUNCHES the sweep is itself
# a live generation on a self-served engine, and its stream closes only after the
# launch tool call returned — an instant idle check can never pass from inside one.
IDLE_WAIT="${IDLE_WAIT:-180}"
run0=$(metric 'vllm:num_requests_running'); t=0
while [[ "$run0" -gt 0 ]]; do
  [[ "$t" -ge "$IDLE_WAIT" ]] && fail "engine not idle after ${IDLE_WAIT}s (running=$run0)"
  sleep 10; t=$((t+10)); run0=$(metric 'vllm:num_requests_running')
done

preempt0=$(metric 'vllm:num_preemptions_total')
printf '%s guard: start idle, preemptions=%s, allow=%s, max=%s\n' \
  "$(date -u +%H:%M:%S)" "$preempt0" "$ALLOW_IPS" "$EXPECTED_MAX" >> "$REPORT"

"${CMD[@]}" & CHILD=$!
while kill -0 "$CHILD" 2>/dev/null; do
  sleep "$POLL"
  foreign=$(docker logs --since "${POLL}s" "$CONTAINER" 2>&1 \
    | grep -E '"(POST|GET)[^"]*(completions|chat|messages)' \
    | grep -Ev "(${ip_grep})[:0-9]" || true)
  [[ -n "$foreign" ]] && fail "non-allowlisted client hit the engine: ${foreign:0:180}"
  run=$(metric 'vllm:num_requests_running')
  [[ "$run" -gt "$EXPECTED_MAX" ]] && fail "num_requests_running=$run > EXPECTED_MAX=$EXPECTED_MAX"
  p=$(metric 'vllm:num_preemptions_total')
  if [[ "$p" -gt "$preempt0" && -z "${PRE_WARNED:-}" ]]; then
    PRE_WARNED=1
    printf '%s NOTE: preemptions %s -> %s (benign self-preemption expected in deep c4 cells)\n' \
      "$(date -u +%H:%M:%S)" "$preempt0" "$p" >> "$REPORT"
  fi
done

rc=0; wait "$CHILD" || rc=$?
printf '%s VERDICT: PASS (wrapped exit %s)\n' "$(date -u +%H:%M:%S)" "$rc" >> "$REPORT"
exit "$rc"
