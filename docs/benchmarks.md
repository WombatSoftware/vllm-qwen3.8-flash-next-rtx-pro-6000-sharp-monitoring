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
| Engine | vLLM nightly `18f8f96025` (the pin in `compose.yaml`), bare upstream image, no overlay |
| Tool | llama-benchy 0.4.0, 3 runs per cell, under `scripts/bench-guard.sh` |
| Date | 2026-10-04 (guard verdict PASS, one self-preemption note, zero restarts) |

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
Measured generation latency: **74.05 ms**.

| test | t/s (total) |
|---|---:|
| pp2048 (c1) | 20,457.64 ±449.75 |
| tg128 (c1) | 156.06 ±17.52 |
| pp2048 (c4) | 12,875.31 ±76.09 |
| tg128 (c4) | 400.03 ±11.85 |
| ctx_pp @ d8192 (c1) | 13,485.14 ±40.19 |
| ctx_tg @ d8192 (c1) | 154.50 ±11.77 |
| pp2048 @ d8192 (c1) | 3,886.74 ±16.61 |
| tg128 @ d8192 (c1) | 149.56 ±6.74 |
| ctx_pp @ d8192 (c4) | 13,130.86 ±81.91 |
| ctx_tg @ d8192 (c4) | 256.33 ±30.10 |
| pp2048 @ d8192 (c4) | 3,711.19 ±27.99 |
| tg128 @ d8192 (c4) | 363.01 ±9.49 |
| ctx_pp @ d32768 (c1) | 12,600.68 ±25.43 |
| ctx_tg @ d32768 (c1) | 133.35 ±16.13 |
| pp2048 @ d32768 (c1) | 4,288.20 ±60.40 |
| tg128 @ d32768 (c1) | 166.14 ±8.55 |
| ctx_pp @ d32768 (c4) | 12,125.20 ±73.67 |
| ctx_tg @ d32768 (c4) | 59.56 ±7.98 |
| pp2048 @ d32768 (c4) | 3,889.87 ±20.76 |
| tg128 @ d32768 (c4) | 274.24 ±17.10 |
| ctx_pp @ d131072 (c1) | 11,539.02 ±24.20 |
| ctx_tg @ d131072 (c1) | 143.55 ±11.39 |
| pp2048 @ d131072 (c1) | 2,248.69 ±11.93 |
| tg128 @ d131072 (c1) | 155.08 ±13.39 |
| ctx_pp @ d131072 (c4) | 10,533.37 ±10.77 |
| ctx_tg @ d131072 (c4) | 14.45 ±0.36 |
| pp2048 @ d131072 (c4) | 176.62 ±32.80 |
| tg128 @ d131072 (c4) | 15.86 ±2.75 |

### How to read this

- **Single-stream decode holds ~133-166 t/s from 0 to 131k context.** That
  flatness across depth is the point of the config; `tg128 @ d131072 (c1)` at
  155.08 is within noise of the shallow figure.
- **Four-way concurrency holds to 32k** (`tg128 @ d32768 (c4)` 274.24) and
  then falls off a cliff at 131k (`15.86`). Four 131k requests need 524,288 KV
  tokens against a 341,041-token pool, so they cannot coexist and the scheduler
  preempts. That is arithmetic, not a fault, and it is why the guard reports
  one preemption per sweep as a note rather than a failure.
- **Context loading is ~10-13k t/s at every depth**, including 4-way at 131k
  (`ctx_pp @ d131072 (c4)` 10,533).
- **Prefill on top of a loaded context is the weak spot.** `pp2048 @ dN (c1)`
  runs 2,249-4,288 t/s. These are the cells that pay for fp8 KV; see below.

## What changed since the 2026-09-12 config

The first published version of this recipe ran `--kv-cache-dtype auto` with an
8192/4096 batch pair on nightly `eed1f3d0` plus a local overlay. Three things
moved, each on its own guarded sweep.

**`--kv-cache-dtype fp8`** (2026-09-17, needs PR #55557). The pool-bound cells
move hugely and single-stream rows stay within noise:

| test | auto | fp8 |
|---|---:|---:|
| ctx_tg @ d131072 (c4) | 3.78 | 13.20 |
| tg128 @ d131072 (c4) | 6.91 | 11.67 |
| ctx_pp @ d131072 (c4) | 6,244 | 10,874 |
| pp2048 @ d131072 (c4) | 101 | 183 |

It is not free. The QSA-compute-bound cells pay for the fp8 kernel:
`tg128 @ d32768 (c4)` went 363.7 -> 244.7 and `pp2048 @ dN (c1)` dropped
20-36%. Later bases recovered part of the first (274.24 above); the second is
still visible in the table. **This is the regression this config accepts**: it trades prefill on
top of deep context for a 4-way deep-context path that is slow instead of
dead. If your workload is single-stream with long incremental prompts, measure
`auto` for yourself (and take the 8192/4096 batch pair with it; see below).

**Batch pair 16384/8192** (2026-09-24). Against 8192/4096 on the same base:
every depth-c1 prefill row +2.4-8.3% (`pp2048 (c1)` 17,556 -> 19,006) and
mid-depth c4 decode +28-61% (`tg128 @ d8192 (c4)` 218 -> 352), nothing lost
outside the preemption-dominated d131k-c4 cells. The cost is KV pool, 468,296
-> 340,041 tokens, which only matters for 4-way at 131k, and that wanted 524k
and fit in neither.

**`--per-request-spec-decode-metrics summary`** (2026-09-24). It was on for the
sweep that shipped the batch pair, and that sweep still beat its predecessor
across the table, so its cost sits inside the noise.

## Engine base

The image is a bare upstream nightly now. The eagle-gate overlay earlier
versions built locally was retired when PR #55390 replaced the model-type gate
with positional draft-group annotation. Proof that prefix caching survives
without the overlay: 406,400 of 626,939 prefix-cache queries hit (64.8%) on an
unpatched image with MTP-3, QSA and fp8 KV all active.

Base moves that showed up in the numbers:

- `ddd6fbca` (09-26): `tg128 @ d131072 (c4)` 15.73 -> 19.54. The next sweep
  put it back at 15.79, so treat that cell as 16-20 with wide run-to-run
  spread, not as a win.
- `36768d1b` (09-29): every depth-c1 prefill row +2.4-5.8%. The `d32768 (c4)`
  rows did not move, which exonerates PR #57105 as their cause.
- `ac68c308` (09-30): no loss beyond 1.6 sigma, `pp2048 (c1)` +4.2%. The c1
  decode rows came in 2-8% lower, all same-signed (`tg128 (c1)` 164 -> 150 at
  1.2 sigma), which looked like a possible regression.
- `0cbac6cd` (10-02): **never booted.** PR #57387 moved the model to the
  upstream transformers config class, which renames a layer type the
  dispatchers did not yet accept; every boot died with
  `ValueError: Invalid layer_type indexed_attention` (vllm#59756). Fixed by
  PR #59621, which is the floor for any future pin. `compose.yaml` carries the
  one-line check.
- `18f8f96025` (10-04): the current pin and the table above. Healthy at
  +150 s with zero restarts, pool 341,041 / 1.30x. No row lost beyond noise against
  `ac68c308`, and the deep prefill rows gained 1-3% with tight spread
  (`ctx_pp @ d131072 (c1)` 11,338 -> 11,539, `pp2048 @ d131072 (c1)` 2,182 ->
  2,249). The c1 decode rows are now mixed-sign against it (-7% to +9%,
  `tg128 (c1)` 150 -> 156), so the same-signed dip one base earlier reads as
  run-to-run spread, not a regression. `tg128 @ d131072 (c4)` at 15.86 stays in
  its 16-20 band.

## Speculative decoding acceptance

Common guidance for this model is 2 speculative tokens, on the grounds that
acceptance length sits around 1.9-2.2 and the third draft token rarely lands.
That does not match what this setup measures. With
`--per-request-spec-decode-metrics detailed` at MTP-3 (2026-09-12):

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
this repo ships `num_speculative_tokens: 3`. A later and much larger sample
(2026-09-26, about 7,400 drafts) measured 2.74. Reproduce with
`./scripts/spec-decode-probe.sh`.

**The trade:** MTP-3 buys roughly +16% single-stream decode over MTP-2 and
costs 10-35% on the concurrent context-load rows (`ctx_tg @ dN (c4)`). Set
`num_speculative_tokens` back to `2` if heavy 4-way fan-out matters more to you
than single-stream latency. This is the one knob most worth reconsidering for
your own workload.

**Check that MTP is actually on.** This deployment once ran for a week without
`--speculative-config` and without the chat template after an edit ate both
lines. Nothing failed: the checkpoint's own template still renders valid tool
calls, and ~90 t/s single-stream looked like kernel noise. What gave it away
was `speculative_config=None` in the engine's init line. After any edit near
`command:`, compare the flag list against the `non-default args` boot line.

## Hard constraints on this GPU

**The KV pool is 1.30x one max-length request.** A healthy boot reports
`GPU KV cache size: 341,041 tokens, Maximum concurrency for 262,144 tokens per
request: 1.30x`. Any change that reserves more GPU memory needs a boot check
before anything else. The pool figure is only comparable between boots with
the same speculative config, because the draft groups claim real pool: an
MTP-less boot of this config reports 983,744 tokens.

The 16384 batch budget depends on fp8 KV. With `--kv-cache-dtype auto` it
leaves 5.61 GiB against the 7.41 GiB one max-length sequence needs and the
engine crash-loops. Change the two together or not at all.

**The first boot can lose an OOM lottery.** At 0.96 utilisation the fused-MoE
autotune occasionally fails its first allocation (1.60 GiB wanted, 1.22 GiB
free; seen on `ac68c308`). `restart: always` retries and the second boot was
clean. If it ever loops, step `--gpu-memory-utilization` down to 0.955.

**Admission control does not do what it sounds like.** `--max-num-queued-reqs`
and `--max-num-queued-tokens` reject with HTTP 503; they do not queue or shape.
They are a load-shedding valve for a multi-instance deployment behind a load
balancer, not a latency knob for a single box. They are also invisible to
benchmark harnesses: for streaming requests vLLM commits HTTP 200 and delivers
the 503 as an in-band error chunk, which tools record as a zero-token success,
silently turning a 4-way row into a 2-way one with *inflated* aggregate
throughput. Do not benchmark with these enabled.

## Benchmark isolation

A sweep is only evidence if nothing else used the engine while it ran. Two
sweeps here were contaminated by an agent client on the LAN, the nightly was
blamed, and an upstream issue was filed and then retracted (vllm#56815). The
client turned out to be the agent session driving the benchmark: its requests
completed inside the cells that looked sick. With no foreign traffic the same
build measured normally.

What came out of it:

- `scripts/bench-guard.sh` wraps the harness. It waits for the engine to go
  idle, then aborts with exit 42 on any request from a client outside the
  allowlist or on concurrency above the bench's own maximum. Preemptions are
  logged as a note, since the deep 4-way cells preempt by arithmetic.
  `./scripts/benchmark.sh` runs under it.
- If the engine also serves agents, run the sweep only when none of them are
  active. The guard will abort a contaminated run, but it cannot make the
  engine quiet.
- A concurrency-1 cell with a standard deviation several times its usual size
  is contention until proven otherwise. Look at `vllm:num_requests_running` and
  the access log before looking at the build.

## Caveats

- Three runs per cell. Small differences do not clear noise.
- Prefix caching is on for all runs, which is how this recipe is meant to be
  served, but it makes the `ctx_*` rows sensitive to run ordering.
- The comparison figures in "What changed" come from sweeps on different
  nightlies than the main table. Read them as deltas, not as absolutes.
