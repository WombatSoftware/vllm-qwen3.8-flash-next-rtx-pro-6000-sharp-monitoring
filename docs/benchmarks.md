# Benchmarks

Measured results for the configuration this repository ships. Reproduce with
`./scripts/benchmark.sh`.

These are single-box numbers for one GPU, one checkpoint and one driver
version. Treat them as evidence for why this config is shaped the way it is,
not as a claim about Qwen3.8-Flash-Next in general.

## Test machine

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 96 GB |
| Driver | 610.57.04, CUDA UMD 13.3 |
| Host RAM | 244 GiB (the PLE table needs ~48 GiB pinned) |
| Checkpoint | `dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix` |
| Engine | vLLM nightly `eed1f3d0` + the eagle-gate overlay in `./Dockerfile` |
| Tool | llama-benchy 0.4.0, 3 runs per cell |
| Date | 2026-09-12 |

Note the **Max-Q**: this card is power-capped at 300 W. A full-power
RTX PRO 6000 will post higher numbers.

## Command

```
llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix \
  --served-model-name qwen-3-8-flash-next \
  --pp 2048 --tg 128 --depth 0 8192 32768 131072 --concurrency 1 4 --runs 3 \
  --latency-mode generation --enable-prefix-caching \
  --format md
```

Reading the labels: `pp` is prefill, `tg` is token generation, `d<N>` is the
context depth the request sits at, `c1`/`c4` is concurrency. `ctx_pp` / `ctx_tg`
are the context-loading phase; plain `pp2048` / `tg128` are the measured phase
on top of that loaded context.

## Results

Aggregate throughput, mean of 3 runs with run-to-run standard deviation.
Measured generation latency: **78.54 ms**.

| test | t/s (total) |
|---|---:|
| pp2048 (c1) | 18,800.44 ±109.80 |
| tg128 (c1) | 169.02 ±11.45 |
| pp2048 (c4) | 12,395.39 ±101.63 |
| tg128 (c4) | 406.07 ±22.60 |
| ctx_pp @ d8192 (c1) | 12,225.49 ±24.16 |
| ctx_tg @ d8192 (c1) | 147.24 ±7.47 |
| pp2048 @ d8192 (c1) | 6,344.96 ±47.03 |
| tg128 @ d8192 (c1) | 163.22 ±6.43 |
| ctx_pp @ d8192 (c4) | 11,526.92 ±50.81 |
| ctx_tg @ d8192 (c4) | 180.31 ±11.75 |
| pp2048 @ d8192 (c4) | 6,046.94 ±21.72 |
| tg128 @ d8192 (c4) | 403.51 ±14.11 |
| ctx_pp @ d32768 (c1) | 10,955.27 ±25.77 |
| ctx_tg @ d32768 (c1) | 143.85 ±2.20 |
| pp2048 @ d32768 (c1) | 4,913.74 ±35.11 |
| tg128 @ d32768 (c1) | 170.81 ±12.78 |
| ctx_pp @ d32768 (c4) | 11,310.04 ±25.70 |
| ctx_tg @ d32768 (c4) | 79.23 ±11.92 |
| pp2048 @ d32768 (c4) | 4,868.58 ±11.53 |
| tg128 @ d32768 (c4) | 363.71 ±5.59 |
| ctx_pp @ d131072 (c1) | 10,391.78 ±6.91 |
| ctx_tg @ d131072 (c1) | 131.34 ±5.91 |
| pp2048 @ d131072 (c1) | 3,205.69 ±32.62 |
| tg128 @ d131072 (c1) | 158.25 ±5.54 |
| ctx_pp @ d131072 (c4) | 6,244.01 ±23.56 |
| ctx_tg @ d131072 (c4) | 3.78 ±0.63 |
| pp2048 @ d131072 (c4) | 101.16 ±6.33 |
| tg128 @ d131072 (c4) | 6.91 ±0.74 |

### How to read this

- **Single-stream decode holds ~131-171 t/s from 0 to 131k context.** That
  flatness across depth is the point of the config; `tg128 @ d131072 (c1)` at
  158.25 is within noise of the shallow figure.
- **Four-way concurrency scales well to 32k** (`tg128 @ d32768 (c4)` 363.71)
  and then falls off a cliff at 131k (`6.91`). Four 131k requests need 524,288
  KV tokens against a 269,228-token pool, so they cannot coexist and the
  scheduler thrashes. This is a KV budget limit, not compute. If you need 4-way
  at 131k, that is the wall to attack -- `--kv-cache-dtype fp8` is the obvious
  next lever and has not been benchmarked here yet.
- **Shallow single-stream prefill is the weak spot** of this config, at
  18,800 t/s. Enabling speculative decoding costs prefill throughput, because
  the draft head is pure overhead while prefilling. If your workload is short
  prompts at concurrency 1, see the MTP note below.

## Speculative decoding acceptance

Common guidance for this model is 2 speculative tokens, on the grounds that
acceptance length sits around 1.9-2.2 and the third draft token rarely lands.
That does not match what this setup measures. With
`--per-request-spec-decode-metrics detailed` at MTP-3:

| metric | value |
|---|---|
| mean acceptance length | **2.695** |
| draft acceptance rate | 0.565 (161 of 285) |
| verify steps sampled | 95 |

Accepted draft tokens per verify step:

| accepted | steps | share |
|---:|---:|---:|
| 0 | 22 | 23.2% |
| 1 | 19 | 20.0% |
| 2 | 20 | 21.1% |
| **3** | **34** | **35.8%** |

All three draft tokens landing is the single most common outcome, which is why
this repo ships `num_speculative_tokens: 3`. Reproduce with
`./scripts/spec-decode-probe.sh` (the flag is not on by default -- it is an
experimental response field, so add it temporarily to measure).

**The trade:** MTP-3 buys roughly +16% single-stream decode over MTP-2 and
costs 10-35% on the concurrent context-load rows (`ctx_tg @ dN (c4)`). Set
`num_speculative_tokens` back to `2` if heavy 4-way fan-out matters more to you
than single-stream latency. This is the one knob most worth reconsidering for
your own workload.

## Hard constraints on this GPU

Two limits are worth knowing before you tune anything:

**The KV pool has almost no headroom.** A healthy boot reports
`GPU KV cache size: 269,228 tokens, Maximum concurrency for 262,144 tokens per
request: 1.03x`. Assume anything that reserves additional GPU memory will fail
to start. `--max-num-batched-tokens 16384` was tried and the engine refuses to
boot outright: the larger activation workspace leaves 5.61 GiB against the
7.41 GiB one max-length sequence needs, and it crash-loops.

**Admission control does not do what it sounds like.** `--max-num-queued-reqs`
and `--max-num-queued-tokens` reject with HTTP 503; they do not queue or shape.
They are a load-shedding valve for a multi-instance deployment behind a load
balancer, not a latency knob for a single box. They are also invisible to
benchmark harnesses: for streaming requests vLLM commits HTTP 200 and delivers
the 503 as an in-band error chunk, which tools record as a zero-token success,
silently turning a 4-way row into a 2-way one with *inflated* aggregate
throughput. Do not benchmark with these enabled.

## Caveats

- Three runs per cell. Small differences do not clear noise.
- Prefix caching is on for all runs, which is how this recipe is meant to be
  served, but it makes the `ctx_*` rows sensitive to run ordering.
