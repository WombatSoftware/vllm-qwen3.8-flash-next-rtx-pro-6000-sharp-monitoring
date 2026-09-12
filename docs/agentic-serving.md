# Agentic serving

This recipe is tuned for tool-calling agents, not for chat. That shapes three
flags and a handful of chat-template options.

## What the server exposes

An OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`, served as
`qwen-3-8-flash-next`. There is **no authentication** -- keep it bound to
localhost and tunnel in (see `.env.example`).

The three flags that make it agentic:

```
--enable-auto-tool-choice        # let the model decide when to call a tool
--tool-call-parser qwen3_xml     # parse Qwen's XML tool blocks into tool_calls
--reasoning-parser qwen3         # split thinking out into reasoning_content
```

`--max-num-seqs 12` is the fan-out budget. The model card suggests 2 as a 96 GB
precaution, which serialises concurrent agents into two slots. KV lives in only
a quarter of the layers here, so a larger budget is cheap.

## Tool calling

Standard OpenAI `tools` / `tool_choice`. Tool calls come back in
`choices[].message.tool_calls`, and the model's thinking, when present, in
`choices[].message.reasoning_content` rather than mixed into `content`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-used")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

r = client.chat.completions.create(
    model="qwen-3-8-flash-next",
    messages=[{"role": "user", "content": "What is the weather in Berlin?"}],
    tools=tools,
)
msg = r.choices[0].message
print(msg.reasoning_content)   # thinking, if any
print(msg.tool_calls)          # parsed calls
```

Feed results back as `role: "tool"` messages with the matching
`tool_call_id`, as usual.

## Chat-template options

The Qwen-Sharp template accepts options through `chat_template_kwargs`. These
are what make it worth vendoring over the checkpoint's built-in template.

| option | default | what it does |
|---|---|---|
| `enable_thinking` | `true` | Master switch for reasoning blocks. |
| `reasoning_effort` | `medium` | `none`, `minimal`, `low`, `medium`, `high`. |
| `auto_disable_thinking_with_tools` | `false` | Drop thinking automatically once tools are in play. |
| `preserve_thinking` | `true` | Keep prior turns' reasoning in context. Alias: `preserve_reasoning`. |
| `tool_call_format` | `xml` | `xml` or `json`. **Leave on `xml`** -- see below. |
| `suppress_tool_instructions` | `false` | Omit the injected tool-use preamble when your framework supplies its own. |
| `max_tool_arg_chars` | `0` (off) | Truncate long tool arguments in the rendered prompt. |
| `max_tool_response_chars` | `0` (off) | Truncate long tool results. Useful against context blowout from chatty tools. |
| `terse` | `false` | Tighter output formatting. |
| `add_vision_id` | `false` | Label images/videos when several appear. |

```python
r = client.chat.completions.create(
    model="qwen-3-8-flash-next",
    messages=[...],
    tools=tools,
    extra_body={"chat_template_kwargs": {
        "reasoning_effort": "high",
        "max_tool_response_chars": 4000,
    }},
)
```

> **Do not set `tool_call_format: "json"`.** The server runs
> `--tool-call-parser qwen3_xml`, which expects XML tool blocks. Switching the
> template to JSON output makes the parser miss the calls, and they surface as
> plain text in `content` instead of `tool_calls`. If you want JSON mode, change
> the server's parser to match.

## Turning thinking off

Two ways, and they differ:

```python
# per request, via the template
extra_body={"chat_template_kwargs": {"enable_thinking": False}}

# or let it drop automatically as soon as tools are present
extra_body={"chat_template_kwargs": {"auto_disable_thinking_with_tools": True}}
```

For long agent loops, `preserve_thinking: False` is often the bigger win: it
keeps earlier turns' reasoning out of context, which matters more than the
per-turn saving once a loop runs tens of turns deep.

## Client configuration

Anything that speaks OpenAI works. Point it at the base URL and use any
non-empty API key.

**LiteLLM**

```yaml
model_list:
  - model_name: qwen-3-8-flash-next
    litellm_params:
      model: openai/qwen-3-8-flash-next
      api_base: http://127.0.0.1:8000/v1
      api_key: not-used
```

**Claude Code**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8000
export ANTHROPIC_AUTH_TOKEN=not-used
export ANTHROPIC_MODEL=qwen-3-8-flash-next
```

Needs a translation proxy in front, since Claude Code speaks the Anthropic
Messages API and this server speaks OpenAI. Any OpenAI-to-Anthropic shim works.

**Plain curl**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-3-8-flash-next",
       "messages":[{"role":"user","content":"Hello"}],
       "max_tokens":512}' | jq '.choices[0].message'
```

## Sizing your fan-out

From `docs/benchmarks.md`, on this GPU:

- Four concurrent agents at up to 32k context each is comfortable: 363 t/s
  aggregate, and roughly 90 t/s per agent.
- Four concurrent agents at 131k context is **not viable**: 6.91 t/s aggregate.
  Four 131k requests need 524,288 KV tokens against a 269,228-token pool, so
  they cannot coexist. Cap agent context well below 131k, or run fewer of them.
- Single-stream decode is flat at ~131-171 t/s from 0 to 131k context, so one
  deep agent is fine. It is the *combination* of deep and concurrent that fails.

`--max-num-seqs 12` is an admission budget, not a promise: twelve deep requests
will not fit in KV. It exists so short requests are not blocked behind long
ones.

## Watching agents run

The bundled Grafana dashboard has panels aimed at exactly this: `Requests
running` / `Requests waiting` for fan-out pressure, `KV cache usage` for the
budget above, and `MTP acceptance` plus `MTP acceptance by draft position` for
whether speculative decoding is earning its keep on your traffic. If acceptance
drops well below the 2.695 in the benchmarks, your workload drafts worse than
the test corpus and `num_speculative_tokens: 2` may serve you better.
