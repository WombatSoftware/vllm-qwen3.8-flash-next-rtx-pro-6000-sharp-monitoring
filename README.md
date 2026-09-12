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
- **A patched engine** — a vLLM nightly plus a one-function overlay, because
  the fix is not upstream yet.
- **MTP-3 speculative decoding**, against the common advice of 2, on measured
  acceptance data.
- **FlashInfer autotune on**, against the model card default.
- **Monitoring is not optional** — Prometheus and Grafana come up with the
  engine, with a 24-panel dashboard provisioned.
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

The KV pool ends up at **269,228 tokens = 1.03x headroom** over a single
max-length request. That thin margin is the defining constraint of this box;
see [`docs/benchmarks.md`](docs/benchmarks.md#hard-constraints-on-this-gpu).

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

**2. Build the engine image** on the GPU host:

```bash
docker build -t local/vllm-openai:qwen4-eaglefix-eed1f3d0 \
  --build-arg BASE=vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657 .
```

The build prints `PATCH-OK` and fails loudly if the upstream anchor has
drifted. See [Why a patched image](#why-a-patched-image).

**3. Start everything:**

```bash
docker compose up -d
until curl -sf -o /dev/null http://127.0.0.1:8000/health; do sleep 5; done
```

First boot with empty caches can take ~30 minutes (weight load, FlashInfer JIT,
torch.compile). Subsequent boots are **127-146 s** — the caches are bind-mounted
so they survive recreates.

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
| Speculative decoding | MTP-3, measured 2.695 mean acceptance length |
| Monitoring | Prometheus scraping vLLM, Grafana with a 24-panel dashboard, provisioned |

Single-stream decode is flat at roughly **131-171 t/s from 0 to 131k context**.
Four-way concurrency reaches **363 t/s aggregate at 32k**. Full numbers, and the
one regression this config accepts, are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Documentation

- [`docs/benchmarks.md`](docs/benchmarks.md) — measured results, the MTP
  acceptance data, and the two hard limits of this GPU.
- [`docs/agentic-serving.md`](docs/agentic-serving.md) — tool calling, thinking
  control, chat-template options, client configuration, fan-out sizing.
- [`AGENTS.md`](AGENTS.md) — conventions and hazards for coding agents (and
  humans) changing this repo.
- `compose.yaml` — every non-obvious flag is explained inline with its reason.

## Why a patched image

Two upstream facts force it:

1. **vLLM 0.29.0 cannot load this checkpoint.** It ships the model (PR #53896)
   but has no PLE offload path at all, so the 47.7 GiB FP8 table has nowhere to
   live on a 96 GB card. The UVA pinned-host offload (PR #54371) landed the day
   after the 0.29.0 cut, so a nightly is required.
2. **A one-function fix is still unmerged.** `_is_deepseek_v4_eagle()` in
   `kv_cache_utils.py` gates a positional eagle-group fallback on `deepseek_v4`
   only. The QSA MTP draft in this checkpoint carries a plain
   `FullAttentionSpec`, so without a `qwen4_exp` entry every KV group is marked
   a draft group and **cross-request prefix-cache reuse is silently disabled**.

`Dockerfile` rewrites that one function and asserts the change landed inside it.
A drifted anchor fails the build rather than costing you the prefix cache
months later. To check whether it is fixed upstream:

```bash
curl -s https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_utils.py \
  | grep -n qwen4_exp
```

If that matches, retag to the bare nightly and delete the Dockerfile.

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

Before changing anything else, read the hard constraints — raising
`--max-num-batched-tokens` prevents the engine from starting at all on this GPU,
and the admission-control caps do not do what their names suggest.

## Credits

This recipe is glue. The work is other people's.

- **[Qwen team, Alibaba](https://huggingface.co/Qwen)** — Qwen3.8-Flash-Next,
  and the official BF16 MTP head this checkpoint uses.
- **[vLLM project](https://github.com/vllm-project/vllm)** — the inference
  engine, the PLE UVA offload in PR #54371, and model support in PR #53896.
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
- **[llama-benchy](https://pypi.org/project/llama-benchy/)** — the benchmark
  harness every number here came from.

None of the above endorse this repository.

## Licence

[Apache-2.0](LICENSE) for the configuration, documentation and scripts.

`chat_template.jinja` is redistributed unmodified under its own Apache-2.0
licence from peculiar-ragdoll / froggeric. Model weights, vLLM, Prometheus and
Grafana are **not** redistributed here and remain under their own licences.
See [`NOTICE`](NOTICE) for the full attribution.
