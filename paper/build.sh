#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TEXBIN="/usr/local/texlive/2025/bin/universal-darwin"
if [[ ! -x "$TEXBIN/latexmk" ]]; then
  echo "error: latexmk not found at $TEXBIN/latexmk" >&2
  exit 1
fi

PATH="$TEXBIN:$PATH" "$TEXBIN/latexmk" -pdf -interaction=nonstopmode -file-line-error main.tex
echo "Built: $SCRIPT_DIR/main.pdf"
