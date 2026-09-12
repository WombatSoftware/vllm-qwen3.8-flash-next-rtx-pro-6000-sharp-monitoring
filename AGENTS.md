# Notes for coding agents

Instructions for AI coding agents working in this repository. Humans may find
the hazards section useful too.

## What this repo is

A single-GPU vLLM deployment recipe. It is **configuration, not an
application** -- there is no build, no test suite, and no code to refactor. The
deliverable is `compose.yaml` plus the documentation that justifies it.

The comments in `compose.yaml` are load-bearing. Nearly every flag has a
measured reason recorded next to it in `docs/benchmarks.md`. Do not "tidy" them
away.

## Hard rules

- **Never commit `.env`.** It holds `GRAFANA_ADMIN_PASSWORD` and possibly
  `HF_TOKEN`. It is gitignored; keep it that way. Use `.env.example` for
  documentation.
- **Never `docker compose down`** on a shared box. It takes Prometheus and
  Grafana with it and drops in-flight requests. Recreate the single service:
  `docker compose up -d qwen-3-8-flash-next`.
- **Never change a flag and a benchmark claim in the same commit** without
  re-running the sweep. The numbers in `docs/benchmarks.md` are measurements,
  not estimates.
- **Do not bump the vendored `chat_template.jinja` silently.** It is a verbatim
  Apache-2.0 copy pinned by hash in `NOTICE`. If you bump it, update the hash
  and the commit in both `NOTICE` and `scripts/verify-chat-template.sh`, and
  run that script.

## Before changing compose.yaml

1. Check `docs/benchmarks.md` for whether the change was already tried. Two
   things were benchmarked and rejected: raising `--max-num-batched-tokens`
   (the engine will not boot) and the admission-control caps (they reject with
   503 rather than queue).
2. Remember the KV pool has **1.03x headroom** over one max-length request.
   Anything reserving more GPU memory fails to start. This is the single most
   common way to break this config.
3. Validate before restarting anything:
   `docker compose config >/dev/null && echo ok`

## Restarting the engine

Warm-cache boot is 127-146 s. A cold first boot with empty caches can take
~30 minutes, which is why `start_period` is 1800 s.

```bash
docker compose up -d qwen-3-8-flash-next
until curl -sf -o /dev/null http://127.0.0.1:8000/health; do sleep 5; done
```

If it does not come up, look for a crash loop before assuming slowness:

```bash
docker inspect -f '{{.RestartCount}}' qwen-3-8-flash-next
docker logs --tail 200 qwen-3-8-flash-next
```

`restart: always` combined with a failing config reloads a 47.7 GiB table every
cycle. Catch it early.

## Benchmarking

`./scripts/benchmark.sh <id>` runs the full sweep, 20-25 minutes. Requirements:

- The engine must already be **healthy**. A run against a restarting engine
  produces meaningless numbers; if the container restarted mid-sweep, discard
  the results.
- `llama-benchy` buffers stdout until exit. Poll `pgrep -f llama-benchy` for
  liveness and the `.done` file for completion. **Do not** watch the log for
  growth; it will look hung when it is fine.
- Nothing else may use the GPU during a run. Check with
  `nvidia-smi --query-compute-apps=pid,process_name --format=csv`.

Beware: the harness does **not** abort on HTTP errors. It drops failed requests
from aggregation, so a partly-failing run yields plausible-looking but wrong
numbers rather than an obvious failure. Always check the log for errors and the
server log for rejections before trusting a result.

## Building the image

The image is built on the GPU host, not pulled. `./Dockerfile` applies a
one-function overlay to upstream vLLM and **asserts the patch landed** -- a
drifted anchor fails the build rather than silently disabling prefix-cache
reuse months later. If the build fails on the anchor, re-diff against upstream
rather than loosening the assertion.

Note that `vllm serve --help` requires a visible GPU in this image; it cannot
be run on a CPU-only box. To check whether a flag exists, grep the installed
source instead:

```bash
docker run --rm --entrypoint bash <image> -c \
  'grep -rn "flag_name" /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py'
```

## Style

Documentation here explains *why*, with numbers, and admits what is not known.
Match that. If you cannot measure a claim, do not make it.
