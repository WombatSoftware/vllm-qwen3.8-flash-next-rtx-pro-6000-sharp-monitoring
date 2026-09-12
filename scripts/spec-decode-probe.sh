#!/usr/bin/env bash
# Report measured speculative-decoding acceptance for one request.
#
# Requires the engine to run with:  --per-request-spec-decode-metrics detailed
# (or `summary`). It is NOT enabled by default in compose.yaml -- it is an
# experimental response field, so add it temporarily when you want to measure.
#
# Note the flag takes a value (none|summary|detailed); passing it bare fails.
set -uo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
SERVED="${SERVED:-qwen-3-8-flash-next}"

# Metrics ride on the final usage chunk, which is only emitted when usage
# reporting is on -- hence stream_options.include_usage.
curl -sS -N "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",
       \"messages\":[{\"role\":\"user\",\"content\":\"Explain in about 200 words why speculative decoding does not change the output distribution.\"}],
       \"max_tokens\":256,\"temperature\":0,
       \"stream\":true,\"stream_options\":{\"include_usage\":true}}" \
| python3 -c '
import json,sys
sd=None
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data:"): continue
    body=line[5:].strip()
    if body=="[DONE]": continue
    try: obj=json.loads(body)
    except json.JSONDecodeError: continue
    m=obj.get("metrics") or {}
    if m.get("speculative_decoding"): sd=m["speculative_decoding"]
if not sd:
    print("No speculative_decoding field. Is --per-request-spec-decode-metrics set?")
    raise SystemExit(1)
print(f"mean acceptance length : {sd[\"mean_acceptance_length\"]:.3f}")
print(f"draft acceptance rate  : {sd[\"draft_acceptance_rate\"]:.3f}")
print(f"verify steps           : {sd[\"num_spec_steps\"]}")
hist=sd.get("acceptance_histogram") or []
tot=sum(hist) or 1
print("accepted draft tokens per verify step:")
for j,c in enumerate(hist):
    print(f"  j={j}: {c:5d}  {100.0*c/tot:5.1f}%")
'
