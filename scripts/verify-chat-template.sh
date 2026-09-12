#!/usr/bin/env bash
# Confirm the vendored chat_template.jinja still matches the upstream commit
# recorded in NOTICE. Run this after any template bump.
set -euo pipefail

REPO="peculiar-ragdoll/Qwen-Sharp-Chat-Templates"
PINNED_COMMIT="85461fc118aaf25e7319c7ecf2481f944aac3a32"
PINNED_SHA256="cdff39fb26b60dc90faa292e726655c6b21f62db497846e02e4c4bbab942a84a"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL="$HERE/chat_template.jinja"

echo "pinned : $REPO @ $PINNED_COMMIT"

actual="$(sha256sum "$LOCAL" | cut -d' ' -f1)"
if [ "$actual" != "$PINNED_SHA256" ]; then
  echo "FAIL: local chat_template.jinja does not match the pinned hash."
  echo "  expected $PINNED_SHA256"
  echo "  actual   $actual"
  exit 1
fi
echo "local  : sha256 matches the pin"

url="https://huggingface.co/$REPO/resolve/$PINNED_COMMIT/chat_template.jinja"
if ! upstream="$(curl -fsSL "$url" | sha256sum | cut -d' ' -f1)"; then
  echo "WARN: could not fetch upstream (offline?). Local pin check passed."
  exit 0
fi
if [ "$upstream" != "$PINNED_SHA256" ]; then
  echo "FAIL: upstream at the pinned commit no longer hashes to the pin."
  echo "  upstream $upstream"
  exit 1
fi
echo "upstream: sha256 matches the pin"

latest="$(curl -fsSL "https://huggingface.co/api/models/$REPO" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])' 2>/dev/null || true)"
if [ -n "$latest" ] && [ "$latest" != "$PINNED_COMMIT" ]; then
  echo
  echo "NOTE: upstream HEAD is now $latest (pin is $PINNED_COMMIT)."
  echo "      A newer template exists. Review its changelog before bumping;"
  echo "      the qwen3_xml render path is what this recipe serves."
fi
echo "OK"
