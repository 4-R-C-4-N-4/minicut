#!/usr/bin/env bash
# Fetch the pinned Verus release into .verus/ (once) and verify the core.
#   scripts/verus.sh            verify crates/minicut-core
#   scripts/verus.sh --setup    only download + install the toolchain
set -euo pipefail
VERUS_RELEASE="0.2026.11.22.1ce6a59"
VERUS_TOOLCHAIN="1.100.0-x86_64-unknown-linux-gnu"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/.verus/$VERUS_RELEASE"

if [ ! -x "$DEST/verus-x86-linux/verus" ]; then
  mkdir -p "$DEST"
  url="https://github.com/verus-lang/verus/releases/download/release%2F$VERUS_RELEASE/verus-$VERUS_RELEASE-x86-linux.zip"
  curl -fsSL "$url" -o "$DEST/verus.zip"
  unzip -q -o "$DEST/verus.zip" -d "$DEST"
  rm "$DEST/verus.zip"
fi
rustup toolchain list | grep -q "^$VERUS_TOOLCHAIN" || rustup toolchain install "$VERUS_TOOLCHAIN" --profile minimal
[ "${1:-}" = "--setup" ] && exit 0

export PATH="$DEST/verus-x86-linux:$PATH"
cd "$ROOT/crates/minicut-core"
cargo verus verify
