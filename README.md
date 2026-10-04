# vLLM · Qwen3.8-Flash-Next · RTX PRO 6000 · Sharp · Monitoring

An **opinionated** single-GPU recipe for serving
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with
[vLLM](https://github.com/vllm-project/vllm) on one **NVIDIA RTX PRO 6000
Blackwell**, using the
[Qwen-Sharp chat template](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
and a bundled local **Prometheus + Grafana** stack.

Every non-default flag here was benchmarked rather than guessed, and the
numbers are in [`docs/benchmarks.md`](docs/benchmarks.md).

### Opinionated about what, exactly

This is not a neutral starting point. It makes these calls for you:

- **A specific checkpoint** — `dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix`,
  NVFP4 main weights with the official BF16 MTP head, not the FP8 one.
- **A specific chat template** — the vendored Qwen-Sharp v22.5.0, deliberately
  *not* the one shipped inside the checkpoint.
- **A pinned nightly engine** — a bare upstream vLLM nightly, pinned by full
  commit sha, because the current release is missing kernels and fixes this
  config depends on.
- **fp8 KV cache and a 16384-token batch budget**, which only boot together.
- **MTP-3 speculative decoding**, against the common advice of 2, on measured
  acceptance data.
- **FlashInfer autotune on**, against the model card default.
- **Monitoring is not optional** — Prometheus and Grafana come up with the
  engine, with a 28-panel dashboard provisioned, including GPU power draw.
- **Tuned for tool-calling agents**, not for chat.

If you want a plain vLLM deployment, start from vLLM's own docs instead. If you
have this GPU and want something that works on the first boot, start here.

---

## Requirements

| | |
|---|---|
| GPU | RTX PRO 6000 Blackwell, 96 GB. Other 96 GB Blackwell cards likely work; anything smaller will not. |
| Host RAM | **≥ 64 GB free.** The 47.7 GiB FP8 PLE table is pinned in host memory. Measured on a 244 GiB box. |
| Disk | ~130 GB for the checkpoint. |
| Software | Docker with Compose v2 and the NVIDIA Container Toolkit. |

The KV pool ends up at **341,041 tokens = 1.30x** a single max-length request.
That margin is the defining constraint of this box; see
[`docs/benchmarks.md`](docs/benchmarks.md#hard-constraints-on-this-gpu).

## Quick start

```bash
git clone https://github.com/WombatSoftware/vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring.git
cd vllm-qwen3.8-flash-next-rtx-pro-6000-sharp-monitoring
cp .env.example .env && $EDITOR .env      # set MODELS_DIR and a Grafana password
```

**1. Get the checkpoint** (~126 GiB):

```bash
hf download dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix \
  --local-dir "$MODELS_DIR/dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix"
```

It is resharded into 141 shards specifically to cap load-time RAM. Keep it
outside your HF hub cache; `compose.yaml` mounts it read-only.

**2. Pull the engine image** on the GPU host:

```bash
docker pull vllm/vllm-openai:nightly-18f8f96025b556071eb627076f94df560fbd3a22
```

`compose.yaml` uses `pull_policy: never`, so this step is not optional. The
pin moves only when you move it. See [Why a nightly](#why-a-nightly).

**3. Start everything:**

```bash
docker compose up -d
until curl -sf -o /dev/null http://127.0.0.1:8000/health; do sleep 5; done
```

First boot with empty caches can take ~30 minutes (weight load, FlashInfer JIT,
torch.compile). Subsequent boots take **about 150 s** — the caches are
bind-mounted so they survive recreates.

**4. Check it:**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-3-8-flash-next",
       "messages":[{"role":"user","content":"What is 17 times 23?"}],
       "max_tokens":300}' | jq -r '.choices[0].message.content'
```

Grafana is at <http://127.0.0.1:3000>, Prometheus at <http://127.0.0.1:9090>.

> **Security:** the vLLM endpoint has no authentication and Prometheus has
> none either. Everything binds to `127.0.0.1` by default. To reach it from
> elsewhere, tunnel rather than rebinding:
> `ssh -L 8000:127.0.0.1:8000 -L 3000:127.0.0.1:3000 <gpu-host>`

## What you get

| | |
|---|---|
| Endpoint | OpenAI-compatible, `http://127.0.0.1:8000/v1`, served as `qwen-3-8-flash-next` |
| Context | 262,144 tokens |
| Tool calling | Auto tool choice with Qwen XML tool blocks parsed into `tool_calls` |
| Reasoning | Split into `reasoning_content`, with per-request effort control |
| Speculative decoding | MTP-3, measured 2.695 mean acceptance length, reported per request |
| KV cache | fp8, 341,041-token pool |
| Monitoring | Prometheus scraping vLLM and a GPU exporter, Grafana with a 28-panel dashboard in 8 rows, provisioned |

Single-stream decode is flat at roughly **133-166 t/s from 0 to 131k context**.
Four-way concurrency reaches **274 t/s aggregate at 32k** and 363 at 8k. Full
numbers, and the regression this config accepts, are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Documentation

- [`docs/benchmarks.md`](docs/benchmarks.md) — measured results, what each
  flag change bought, the MTP acceptance data, and the hard limits of this GPU.
- [`docs/agentic-serving.md`](docs/agentic-serving.md) — tool calling, thinking
  control, chat-template options, client configuration, fan-out sizing.
- [`AGENTS.md`](AGENTS.md) — conventions and hazards for coding agents (and
  humans) changing this repo.
- `compose.yaml` — every non-obvious flag is explained inline with its reason.

## Why a nightly

The image is a bare upstream nightly, pinned by full commit sha. No release
will do yet:

- **v0.29.0 cannot load this checkpoint.** It ships the model (PR #53896) but
  has no PLE offload path, so the 47.7 GiB FP8 table has nowhere to live on a
  96 GB card.
- **v0.30.0 can, but predates what this config runs on**: the fp8 QSA kernels
  for this GPU (PR #55557), a tool-call parser fix (PR #56635) and the
  speculative-decoding correctness fixes (PRs #56734, #57885).

Earlier versions of this recipe built a local image with a one-function
overlay on `kv_cache_utils.py`, because without it cross-request prefix-cache
reuse was silently disabled for this model. Upstream replaced that code path
with positional draft-group annotation (PR #55390), the overlay became
unnecessary, and the `Dockerfile` is gone.

**Moving the pin.** Not every nightly boots. `0cbac6cd` (2026-10-02) died on
every start with `ValueError: Invalid layer_type indexed_attention`
(vllm#59756, fixed by PR #59621). Before pinning a newer base, check it
contains the fix:

```bash
docker run --rm --entrypoint bash vllm/vllm-openai:nightly-<sha> -c \
  'grep -rn indexed_attention /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/'
```

Then boot it, confirm the KV pool line and `speculative_config` in the log, and
re-run the sweep before trusting it. Keep the previous image until the sweep
passes.

## The chat template

`chat_template.jinja` is a **verbatim, unmodified copy** of Qwen-Sharp v22.5.0,
pinned by hash. It is vendored so this recipe is self-contained and
reproducible, not because it was changed.

```bash
./scripts/verify-chat-template.sh   # confirms it still matches upstream
```

It is used instead of the template inside the checkpoint because it gives
per-request control over thinking, reasoning effort, tool-call format and tool
output truncation — see
[`docs/agentic-serving.md`](docs/agentic-serving.md#chat-template-options).
The `qwen3_xml` render path is what this server is configured to parse.

## Tuning it for your workload

The knob most worth reconsidering is **`num_speculative_tokens`**. MTP-3 buys
about +16% single-stream decode and costs 10-35% on concurrent context-load
throughput. Drop it to `2` if heavy 4-way fan-out matters more to you than
single-stream latency.

`--kv-cache-dtype fp8` is the other one. It is what makes 4-way deep context
slow instead of dead, and it costs 20-36% on prefill stacked on an
already-loaded context. If you go back to `auto`, take
`--max-num-batched-tokens 8192 --long-prefill-token-threshold 4096` with it:
the 16384 budget does not boot without fp8.

Before changing anything else, read the hard constraints, and note that the
admission-control caps do not do what their names suggest.

If you benchmark a change, use `./scripts/benchmark.sh` (it runs under the
isolation guard) and keep the `.guard` verdict with the results.

## Credits

This recipe is glue. The work is other people's.

- **[Qwen team, Alibaba](https://huggingface.co/Qwen)** — Qwen3.8-Flash-Next,
  and the official BF16 MTP head this checkpoint uses.
- **[vLLM project](https://github.com/vllm-project/vllm)** — the inference
  engine, the PLE UVA offload in PR #54371, model support in PR #53896, and
  the fp8 QSA kernels in PR #55557.
- **[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll)** — the
  [Qwen-Sharp chat templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates),
  vendored here under Apache-2.0. The reason tool calling and thinking control
  behave well.
- **[froggeric](https://huggingface.co/froggeric)** —
  [Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates),
  the upstream that Qwen-Sharp builds on. The template still carries the
  `qwen3.8-froggeric` lineage in its version string.
- **[dicksondickson](https://huggingface.co/dicksondickson)** — the
  [resharded MTP-fix checkpoint](https://huggingface.co/dicksondickson/Qwen3.8-Flash-Next-NVFP4-reshard-mtp-fix)
  this recipe serves, which swaps in the official BF16 MTP head and reshards to
  cap load-time RAM.
- **[NVIDIA](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** — the
  NVFP4 quantisation the checkpoint is built from.
- **[Prometheus](https://prometheus.io/)** and
  **[Grafana](https://grafana.com/)** — the monitoring stack.
- **[nvidia_gpu_exporter](https://github.com/utkuozdemir/nvidia_gpu_exporter)**
  — GPU power, temperature and VRAM metrics.
- **[llama-benchy](https://pypi.org/project/llama-benchy/)** — the benchmark
  harness every number here came from.

None of the above endorse this repository.

## Licence

[Apache-2.0](LICENSE) for the configuration, documentation and scripts.

`chat_template.jinja` is redistributed unmodified under its own Apache-2.0
licence from peculiar-ragdoll / froggeric. Model weights, vLLM, Prometheus,
Grafana and nvidia_gpu_exporter are **not** redistributed here and remain under
their own licences. See [`NOTICE`](NOTICE) for the full attribution.
