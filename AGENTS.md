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

There is no `Dockerfile` any more. The engine is a bare upstream nightly pinned
by full-sha tag in `compose.yaml`.

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
- **Never run a sweep outside `scripts/bench-guard.sh`**, and never while an
  agent session is using this engine. Read "Benchmark isolation" in
  `docs/benchmarks.md` first. If the engine is the one serving *you*, your own
  turns are the interference.
- **After any edit near `command:` in `compose.yaml`, re-check the flag list.**
  An edit to the comments once deleted `--speculative-config` and
  `--chat-template` and the engine ran for a week without them, passing every
  smoke test. Compare against the `non-default args` line of the next boot;
  do not eyeball the diff.
- **Do not bump the vendored `chat_template.jinja` silently.** It is a verbatim
  Apache-2.0 copy pinned by hash in `NOTICE`. If you bump it, update the hash
  and the commit in both `NOTICE` and `scripts/verify-chat-template.sh`, and
  run that script.

## Before changing compose.yaml

1. Check `docs/benchmarks.md` for whether the change was already tried. The
   admission-control caps were benchmarked and rejected (they reject with 503
   rather than queue). `--max-num-batched-tokens 16384` and
   `--kv-cache-dtype fp8` are a pair: the first does not boot without the
   second.
2. The KV pool is **341,041 tokens, 1.30x** one max-length request. Anything
   reserving more GPU memory needs a boot check before anything else. The pool
   line is only comparable between boots with the same speculative config.
3. Validate before restarting anything:
   `docker compose config >/dev/null && echo ok`

## Restarting the engine

Warm-cache boot is about 150 s. A cold first boot with empty caches can take
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
cycle. Catch it early. One restart is not yet a loop: the first boot can lose
a CUDA-OOM lottery in the fused-MoE autotune and come up clean on the retry.
Three is a loop.

Then confirm the boot is the config you meant, from the log:

```bash
docker logs qwen-3-8-flash-next 2>&1 | grep -E \
  "non-default args|GPU KV cache size" | cut -c1-900
```

Expect `speculative_config` with `num_speculative_tokens: 3`, `kv_cache_dtype:
fp8`, the mounted chat template, and a pool of about 341k tokens.

## Benchmarking

`./scripts/benchmark.sh <id>` runs the full sweep under the isolation guard,
20-25 minutes. If the engine also serves you, launch it detached and say
nothing until the `.done` file exists: the guard waits up to three minutes for
the engine to go idle, and any turn you take after that voids the run.
Requirements:

- The engine must already be **healthy**. A run against a restarting engine
  produces meaningless numbers; if the container restarted mid-sweep, discard
  the results.
- `llama-benchy` buffers stdout until exit. Poll `pgrep -f llama-benchy` for
  liveness and the `.done` file for completion. **Do not** watch the log for
  growth; it will look hung when it is fine.
- Nothing else may use the GPU during a run. Check with
  `nvidia-smi --query-compute-apps=pid,process_name --format=csv`.
- Nothing else may use the **engine** during a run. The guard enforces this:
  exit 42 and `VERDICT: FAIL` mean the numbers are void, not that the build is
  slow. Its allowlist is 127.0.0.1 plus the compose network's gateway, which is
  how host-side requests appear in the access log.
- A `NOTE: preemptions` line is expected. The 4-way 131k cells need more KV
  than the pool holds and preempt by arithmetic.
- A concurrency-1 cell with a wildly inflated standard deviation is contention
  until proven otherwise.

Beware: the harness does **not** abort on HTTP errors. It drops failed requests
from aggregation, so a partly-failing run yields plausible-looking but wrong
numbers rather than an obvious failure. Always check the log for errors and the
server log for rejections before trusting a result.

## Moving the image pin

The image is pulled, not built. `pull_policy: never` plus a full-sha tag means
the base only moves when `compose.yaml` does.

1. Pull the new `vllm/vllm-openai:nightly-<full sha>` tag.
2. Check it contains PR #59621 (README "Why a nightly" has the grep). Bases
   without it crash before weight load.
3. Keep the old image. Recreate the engine, then verify the boot lines above
   and that prefix caching still hits (`vllm:prefix_cache_hits_total` must rise
   across two requests sharing a prefix; do not rely on
   `usage.prompt_tokens_details`, which is null on current bases).
4. Run a guarded sweep and compare against `docs/benchmarks.md` before calling
   the new base the baseline. Delete the old image only after that.

When blaming a nightly for a regression, show that a suspect commit touches a
file this model's runtime path imports. Commit-title plausibility is how the
retracted upstream issue got filed.

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
