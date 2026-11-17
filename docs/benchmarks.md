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
| Driver | 615.71.09 (CUDA 13; v0.33.0 and v0.34.0 images are CUDA 13 only) |
| Host RAM | 122 GiB on 2026-11-17 (the PLE table needs ~48 GiB pinned); 244 GiB when the 2026-11-02 column was measured |
| Checkpoint | `dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix` |
| Engine | vLLM v0.34.0 (the digest pin in `compose.yaml`), bare upstream image, no overlay |
| Tool | llama-benchy 0.4.0, 3 runs per cell, under `scripts/bench-guard.sh` |
| Date | 2026-11-17 (guard verdict PASS, one self-preemption note, zero restarts, 407 requests all HTTP 200) |
| Box state | The card under test was one of four. The other three held loaded, idle engines during both 2026-11-17 sweeps |

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

Aggregate throughput (t/s, total), mean of 3 runs with run-to-run standard
deviation. Measured generation latency on v0.34.0: **56.52 ms**.

Three columns, because the pin and one flag moved together:

- **v0.34.0 (shipped)**: the configuration in `compose.yaml`, 2026-11-17.
- **v0.33.0, same day**: the previous pin and flag list (no
  `--attention-config.indexer_kv_dtype`), swept 15 minutes later on the same
  card in the same box state. This is the control; "change" is against it.
- **v0.33.0, 2026-11-02**: the table this page carried until now.

| test | v0.34.0 (shipped) | v0.33.0, same day | change | v0.33.0, 2026-11-02 |
|---|---:|---:|---:|---:|
| pp2048 (c1) | 12,727.70 ±2,547.67 | 11,176.56 ±1,389.66 | +13.9% | 20,419.09 ±548.16 |
| tg128 (c1) | 160.45 ±8.55 | 161.67 ±9.50 | -0.8% | 159.89 ±8.12 |
| pp2048 (c4) | 12,382.77 ±122.14 | 12,299.71 ±248.28 | +0.7% | 11,664.63 ±1,692.91 |
| tg128 (c4) | 396.12 ±6.80 | 368.91 ±16.37 | +7.4% | 413.33 ±19.89 |
| ctx_pp @ d8192 (c1) | 13,560.07 ±63.21 | 13,782.57 ±190.97 | -1.6% | 14,687.27 ±63.33 |
| ctx_tg @ d8192 (c1) | 130.94 ±14.23 | 135.78 ±5.99 | -3.6% | 140.88 ±10.03 |
| pp2048 @ d8192 (c1) | 7,260.73 ±64.17 | 7,332.49 ±173.45 | -1.0% | 7,948.43 ±33.49 |
| tg128 @ d8192 (c1) | 159.12 ±10.54 | 160.67 ±11.20 | -1.0% | 162.32 ±6.13 |
| ctx_pp @ d8192 (c4) | 12,546.68 ±18.44 | 12,520.94 ±41.36 | +0.2% | 12,816.66 ±214.44 |
| ctx_tg @ d8192 (c4) | 190.62 ±25.64 | 171.48 ±5.52 | +11.2% | 172.84 ±1.65 |
| pp2048 @ d8192 (c4) | 6,594.71 ±25.24 | 6,533.41 ±30.05 | +0.9% | 6,684.01 ±83.03 |
| tg128 @ d8192 (c4) | 343.89 ±19.06 | 365.15 ±5.15 | -5.8% | 347.42 ±21.14 |
| ctx_pp @ d32768 (c1) | 12,415.90 ±18.77 | 12,312.72 ±15.94 | +0.8% | 12,928.95 ±16.53 |
| ctx_tg @ d32768 (c1) | 140.25 ±7.40 | 135.98 ±12.78 | +3.1% | 138.63 ±4.44 |
| pp2048 @ d32768 (c1) | 8,688.05 ±171.75 | 8,714.32 ±153.44 | -0.3% | 8,686.73 ±107.77 |
| tg128 @ d32768 (c1) | 172.58 ±11.83 | 160.61 ±2.33 | +7.5% | 164.08 ±3.08 |
| ctx_pp @ d32768 (c4) | 12,325.67 ±8.63 | 12,240.52 ±22.99 | +0.7% | 12,732.60 ±23.56 |
| ctx_tg @ d32768 (c4) | 63.77 ±13.15 | 76.25 ±6.81 | -16.4% | 61.43 ±7.42 |
| pp2048 @ d32768 (c4) | 8,240.75 ±50.16 | 8,155.08 ±13.23 | +1.1% | 8,364.80 ±30.80 |
| tg128 @ d32768 (c4) | 361.79 ±3.97 | 379.41 ±24.70 | -4.6% | 358.19 ±15.46 |
| ctx_pp @ d131072 (c1) | 12,044.14 ±7.01 | 11,821.48 ±43.74 | +1.9% | 12,350.83 ±5.75 |
| ctx_tg @ d131072 (c1) | 124.06 ±12.37 | 155.40 ±28.57 | -20.2% | 144.41 ±6.22 |
| pp2048 @ d131072 (c1) | 3,528.20 ±55.82 | 3,390.94 ±53.27 | +4.0% | 3,503.82 ±49.50 |
| tg128 @ d131072 (c1) | 161.84 ±15.39 | 145.62 ±19.27 | +11.1% | 158.75 ±15.35 |
| ctx_pp @ d131072 (c4) | 9,714.91 ±36.65 | 10,929.78 ±23.35 | -11.1% | 10,721.89 ±800.15 |
| ctx_tg @ d131072 (c4) | 11.60 ±2.18 | 13.16 ±0.95 | -11.9% | 12.81 ±3.00 |
| pp2048 @ d131072 (c4) | 180.21 ±37.56 | 197.85 ±14.63 | -8.9% | 180.55 ±6.28 |
| tg128 @ d131072 (c4) | 11.63 ±2.61 | 12.50 ±0.92 | -7.0% | 11.92 ±0.67 |

**The two days are not comparable.** The same-day v0.33.0 control sits far
below the 2026-11-02 sweep of the same release on shallow prefill
(`pp2048 (c1)` 11,177 against 20,419) and 3-8% below it on the `ctx_pp (c1)`
rows. Known differences on 2026-11-17: half the host RAM, and three other
engines loaded on the other cards (what the other cards held on 2026-11-02
was not recorded). Which difference costs the prefill was not isolated. Read "change" for what the release did, and the 2026-11-02
column as a record of a faster box state.

### How to read this

- **v0.34.0 with the indexer flag is a wash against v0.33.0 on the same day.**
  No row loses more than 3 sigma except `ctx_pp @ d131072 (c4)` (-11.1%), a
  preemption-dominated cell whose arithmetic moved with the pool (438,539
  tokens against 401,790), so it is not comparable across the two. The c1
  context-load rows at 32k and 131k gain 0.8% and 1.9% with tight spread.
- **Single-stream decode holds ~124-173 t/s from 0 to 131k context.** That
  flatness across depth is the point of the config; `tg128 @ d131072 (c1)` at
  161.84 is within noise of the shallow figure.
- **Four-way concurrency holds to 32k** (`tg128 @ d32768 (c4)` 361.79) and
  then falls off a cliff at 131k (`11.63`). Four 131k requests need 524,288 KV
  tokens against a 438,539-token pool, so they cannot coexist and the
  scheduler preempts. That is arithmetic, not a fault, and it is why the guard
  reports a preemption note per sweep rather than a failure.
- **Context loading is ~9.7-13.6k t/s at every depth**, including 4-way at
  131k (`ctx_pp @ d131072 (c4)` 9,715).
- **Prefill on top of a loaded context is still the weakest row family**:
  `pp2048 @ dN (c1)` runs 3,528-8,688 t/s. v0.33.0 roughly doubled it from
  2,249-4,288 on `18f8f96025`, and v0.34.0 leaves it where it was. These are
  the cells that pay for fp8 KV; see below.
- **`pp2048 (c1)` is noisy on both 2026-11-17 sweeps** (standard deviation 12%
  and 20% of the mean). Do not read its +13.9% as a gain.

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
20-36%. Later bases recovered most of the first (358-362 above), and v0.33.0
roughly doubled the second family, but `auto` has not been re-measured on
v0.33.0, so whether fp8 still costs prefill at depth there is unknown.
**This is the trade this config accepts**: possibly slower prefill on top of
deep context for a 4-way deep-context path that is slow instead of dead. If
your workload is single-stream with long incremental prompts, measure `auto`
for yourself (and take the 8192/4096 batch pair with it; see below).

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

The image is a bare upstream release now, pinned by digest; earlier pins were
full-sha nightlies. The eagle-gate overlay earlier
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
- `18f8f96025` (10-04): healthy at +150 s with zero restarts, pool 341,041 /
  1.30x. No row lost beyond noise against `ac68c308`, and the deep prefill
  rows gained 1-3% with tight spread (`ctx_pp @ d131072 (c1)` 11,338 ->
  11,539, `pp2048 @ d131072 (c1)` 2,182 -> 2,249). The c1 decode rows are
  mixed-sign against it (-7% to +9%, `tg128 (c1)` 150 -> 156), so the
  same-signed dip one base earlier reads as run-to-run spread, not a
  regression. `tg128 @ d131072 (c4)` at 15.86 stays in its 16-20 band.
- `21d93d0d8` (10-06): QSA prefill wins exactly where PR #59533 (QSA projection
  merge) aims: `ctx_pp @ d131072 (c1)` +3.8%, `ctx_pp @ d32768 (c1)` +1.6%,
  `ctx_pp @ d32768 (c4)` +2.8%. Pool 348,676 / 1.33x.
- `fe4df3df0e` (10-12): **rolled back.** No wins, and a same-signed loss
  cluster on deep-context prefill c1 (`ctx_pp @ d131072 (c1)` -2.8%,
  `pp2048 @ d32768 (c1)` -3.9%). The suspect, PR #59528, is also in v0.32.0,
  where the cluster fully inverted, so the cause was never isolated.
- `v0.32.0` (10-21): first release pin. No row moved beyond 3 sigma against
  `21d93d0d8`. Pool 358,857 / 1.37x.
- `v0.33.0` (11-02): the current pin and the table above. Healthy at +180 s,
  pool 401,790 / 1.53x. No row lost more than 3 sigma against `v0.32.0`; the
  whole `pp2048 @ dN` family gained 82-110% (`pp2048 @ d8192 (c1)` est. prefill
  time 528 -> 258 ms), and `ctx_pp (c1)` 2-9%. Tail-block reuse under MTP is
  exactly what PR #61589 (lookahead block hashes, now default) advertises, but
  this is one sample. The pool grew 12%, which changes the deep 4-way
  preemption arithmetic: compare those cells only against sweeps on this pool.
- `v0.34.0` (11-17): the current pin and the first column above. Healthy at
  +191 to +231 s over three boots, zero restarts. With the v0.33.0 flag list
  the pool FELL to 291,543 / 1.11x: the release gives the KV cache more
  memory (8.86 GiB against 6.93) but each 3,200-token block costs 76% more,
  because the QSA indexer cache no longer follows `--kv-cache-dtype fp8` for
  this checkpoint. Adding `--attention-config.indexer_kv_dtype fp8` brings it
  to 438,539 / 1.67x (440,989 on another boot), 9% above v0.33.0. That flag
  is in `compose.yaml` now and the sweep ran with it. Against a v0.33.0
  control swept the same day: no row lost beyond 3 sigma outside the
  pool-bound `ctx_pp @ d131072 (c4)`. The flag was not swept on its own, so
  what the indexer dtype costs or buys in isolation is unknown.

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

On v0.34.0 (2026-11-17) six single requests from 2,855 to 261,200 prompt
tokens, temperature 0 and thinking off, measured 2.27-2.40 from `/metrics`
counter deltas. Six requests of one prompt family are a spot check, not a
replacement for the samples above.

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

**The KV pool is 1.67x one max-length request.** A healthy v0.34.0 boot
reports `GPU KV cache size: 438,539 tokens, Maximum concurrency for 262,144
tokens per request: 1.67x`. It only gets there with
`--attention-config.indexer_kv_dtype fp8`; without that flag the same release
reports 291,543 / 1.11x. Any change that reserves more GPU memory needs a
boot check before anything else. The pool also moves with the base image
(341,041 on `18f8f96025`, 358,857 on `v0.32.0`, 401,790 on `v0.33.0`), and
boot-to-boot it varies by about ±1%, so sub-1% deltas mean nothing. It is
only comparable between boots with the same speculative config, because the
draft groups claim real pool: an MTP-less boot on an earlier nightly reported
983,744 tokens.

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
  bases than the main table. Read them as deltas, not as absolutes.
