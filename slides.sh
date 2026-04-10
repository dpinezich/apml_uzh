#!/usr/bin/env bash
# ── APML UZH — Slidev slide launcher ──────────────────────────────────────
# Always run from the project root.
#
# Usage:
#   ./slides.sh 2          → serve Ch02 with hot-reload (http://localhost:3030)
#   ./slides.sh 2 pdf      → export Ch02 → exports/ch02_slides.pdf
#   ./slides.sh 2 build    → build static site → exports/ch02/
#   ./slides.sh            → list all chapters

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
EXPORTS="$ROOT/exports"

CHAPTERS=(
  ""
  "1-introduction/01-slides/ch01_slides.md"
  "2-selection_cleaning_preparing/01-slides/ch02_slides.md"
  "3-supervised_learning/01-slides/ch03_slides.md"
  "3-supervised_learning/01-slides/ch04_slides.md"
  "3-supervised_learning/01-slides/ch05_slides.md"
  "3-supervised_learning/01-slides/ch06_slides.md"
  "4-unsupervised_learning/01-slides/ch07_slides.md"
  "4-unsupervised_learning/01-slides/ch08_slides.md"
  "4-unsupervised_learning/01-slides/ch09_slides.md"
  "5-reinforcement_learning/01-slides/ch10_slides.md"
  "5-reinforcement_learning/01-slides/ch11_slides.md"
  "6-capstone_ml/01-slides/ch12_slides.md"
)

CH="${1:-}"
MODE="${2:-serve}"

cd "$ROOT"

# ── No chapter → list available chapters ─────────────────────────────────
if [ -z "$CH" ]; then
  echo "APML UZH Slides — available chapters:"
  for i in "${!CHAPTERS[@]}"; do
    [ -z "${CHAPTERS[$i]}" ] && continue
    echo "  $i  ${CHAPTERS[$i]}"
  done
  echo ""
  echo "Usage: ./slides.sh <chapter> [serve|pdf|build]"
  echo "  serve  → hot-reload dev server (default)"
  echo "  pdf    → export to PDF"
  echo "  build  → build static site"
  exit 0
fi

FILE="${CHAPTERS[$CH]:-}"
if [ -z "$FILE" ]; then
  echo "Unknown chapter: $CH (use 1–12)"
  exit 1
fi

BASENAME="$(basename "$FILE" .md)"
CHPAD="$(printf '%02d' "$CH")"

case "$MODE" in

  # ── PDF export ────────────────────────────────────────────────────────────
  pdf)
    mkdir -p "$EXPORTS"
    OUT="$EXPORTS/${CHPAD}_slides.pdf"
    echo "Exporting Ch$CH → $OUT"
    npx slidev export "$FILE" \
      --output "$OUT" \
      --format pdf \
      --with-clicks
    echo "Done: $OUT"
    ;;

  # ── Static build ──────────────────────────────────────────────────────────
  build)
    OUT_DIR="$EXPORTS/ch${CHPAD}"
    mkdir -p "$OUT_DIR"
    echo "Building Ch$CH → $OUT_DIR"
    npx slidev build "$FILE" \
      --out "$OUT_DIR" \
      --base "/ch${CHPAD}/"
    echo "Done: $OUT_DIR"
    ;;

  # ── Dev server with hot-reload ────────────────────────────────────────────
  serve|*)
    echo "Serving Ch$CH → http://localhost:3030"
    echo "Press O to open, S for speaker notes."
    npx slidev "$FILE"
    ;;

esac
