# syntax=docker/dockerfile:1
# qwen4_exp eagle-group fallback overlay on upstream nightly.
#
# Base must contain PR #54371 (UVA PLE-offload + Engram TP, merge commit
# 3116c5d0, landed 2026-09-09 after v0.29.0 was cut). v0.29.0 itself has the
# Qwen3.8-Flash-Next model (PR #53896) but NO PLE offload path at all.
#
# Single-function overlay: _is_deepseek_v4_eagle() in
# vllm/v1/core/kv_cache_utils.py gates the positional eagle-group fallback on
# deepseek_v4 only. The QSA MTP draft in the dicksondickson export carries a
# plain FullAttentionSpec, so without the qwen4_exp entry every KV group is
# marked a draft group and cross-request prefix-cache reuse is silently
# disabled (the bug the old 17-file overlay fixed; still unfixed upstream at
# the pinned base).
#
# The gate's exact expression drifts between nightlies (nightly e7edf17c had
# `model_type == "deepseek_v4"`; later main widened it to a
# deepseek_v4/deepseek_v41 tuple). So the overlay is a Python rewrite that
# accepts either form and asserts it landed INSIDE _is_deepseek_v4_eagle --
# a drifted anchor fails the build, not the prefix cache months later.
#
# The old overlay's other two fixes (mtp.py quantized_layers remap, modelopt.py
# FP8_BLOCK_SCALES dispatch) are dropped: they only mattered for the nvidia
# FP8-MTP export and stay inert for the BF16-MTP dickson checkpoint in use.
#
# Build on the GPU host:
#   docker build -t local/vllm-openai:qwen4-eaglefix-eed1f3d0 \
#     --build-arg BASE=vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657 .
# When moving to a newer nightly: re-check
#   curl -s https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/core/kv_cache_utils.py | grep -n qwen4_exp
# If upstream ever lands the entry, retag to the bare nightly and delete this file.

ARG BASE=vllm/vllm-openai:nightly-eed1f3d0c6043bd494424a22443ee198dd56f657
FROM ${BASE}

RUN python3 - <<'PY'
import inspect
p = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py"
src = open(p).read()
eq = 'model_config.hf_config.model_type == "deepseek_v4"'
tup = 'model_config.hf_config.model_type in (\n        "deepseek_v4",\n        "deepseek_v41",\n    )'
if src.count(eq) == 1:
    out = src.replace(eq, 'model_config.hf_config.model_type in ("deepseek_v4", "qwen4_exp")')
elif src.count(tup) == 1:
    out = src.replace(tup, tup.replace('"deepseek_v41",', '"deepseek_v41",\n        "qwen4_exp",'))
else:
    raise SystemExit(f"eagle-gate anchor not found (eq={src.count(eq)}, tup={src.count(tup)}); re-diff against upstream")
open(p, "w").write(out)
from vllm.v1.core import kv_cache_utils as k
s = inspect.getsource(k._is_deepseek_v4_eagle)
assert '"qwen4_exp"' in s, s
print("PATCH-OK")
PY
