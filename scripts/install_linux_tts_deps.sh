#!/usr/bin/env bash
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This helper currently supports apt-based Linux distributions." >&2
  echo "Install espeak-ng with your distribution package manager." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y espeak-ng

echo "Installed Linux TTS dependency: espeak-ng"
