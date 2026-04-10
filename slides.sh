#!/usr/bin/env bash
# ── APML UZH — Slidev slide launcher ──────────────────────────────────────
# Always run from the project root.
#
# Usage:
#   ./slides.sh 2          → serve Ch02 with hot-reload (http://localhost:3030)
#   ./slides.sh 2 fresh    → serve Ch02 with Vite cache cleared
#   ./slides.sh 2 pdf      → export Ch02 → exports/ch02_slides.pdf
#   ./slides.sh 2 build    → build static site → exports/ch02/
#   ./slides.sh flush      → delete Vite cache without serving
#   ./slides.sh            → list all chapters
#
# Slides are always served from the slidev/ directory so that style.css,
# layouts/, and public/ are resolved from real files (no symlink ambiguity).

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
SLIDEV_DIR="$ROOT/slidev"
EXPORTS="$ROOT/exports"
ENTRY="$SLIDEV_DIR/current.md"

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

# ── flush → delete Vite cache ────────────────────────────────────────────
if [ "$CH" = "flush" ]; then
  echo "Flushing Vite cache..."
  rm -rf "$ROOT/node_modules/.vite"
  echo "Done. Run ./slides.sh <chapter> to start fresh."
  exit 0
fi

# ── No chapter → list available chapters ─────────────────────────────────
if [ -z "$CH" ]; then
  echo "APML UZH Slides — available chapters:"
  for i in "${!CHAPTERS[@]}"; do
    [ -z "${CHAPTERS[$i]}" ] && continue
    echo "  $i  ${CHAPTERS[$i]}"
  done
  echo ""
  echo "Usage: ./slides.sh <chapter> [serve|fresh|pdf|build]"
  echo "  serve  → hot-reload dev server (default)"
  echo "  fresh  → serve with Vite cache cleared"
  echo "  pdf    → export to PDF"
  echo "  build  → build static site"
  echo ""
  echo "  flush  → delete Vite cache (./slides.sh flush)"
  exit 0
fi

FILE="${CHAPTERS[$CH]:-}"
if [ -z "$FILE" ]; then
  echo "Unknown chapter: $CH (use 1–12)"
  exit 1
fi

CHPAD="$(printf '%02d' "$CH")"

# ── Wire up this chapter's directory ─────────────────────────────────────
# 1. Symlink current.md → chapter slide file  (sets slidev/ as userRoot)
# 2. Symlink public/chNN/ → chapter 01-slides/ (images served as /chNN/X.png)
CHAPTER_DIR="$(dirname "$ROOT/$FILE")"
PUBLIC_IMG="$SLIDEV_DIR/public/ch${CHPAD}"

cleanup() {
  rm -f "$ENTRY"
  # Leave the public symlink in place so repeated runs skip re-linking
}
trap cleanup EXIT INT TERM

ln -sf "$ROOT/$FILE" "$ENTRY"

# Create/refresh the public image symlink for this chapter
if [ ! -L "$PUBLIC_IMG" ] && [ -d "$PUBLIC_IMG" ]; then
  rm -rf "$PUBLIC_IMG"          # replace old real directory with symlink
fi
ln -sfn "$CHAPTER_DIR" "$PUBLIC_IMG"

cd "$SLIDEV_DIR"

case "$MODE" in

  # ── PDF export ────────────────────────────────────────────────────────────
  pdf)
    mkdir -p "$EXPORTS"
    OUT="$EXPORTS/${CHPAD}_slides.pdf"
    echo "Exporting Ch$CH → $OUT"
    npx --prefix "$ROOT" slidev export "current.md" \
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
    npx --prefix "$ROOT" slidev build "current.md" \
      --out "$OUT_DIR" \
      --base "/ch${CHPAD}/"
    echo "Done: $OUT_DIR"
    ;;

  # ── Dev server with cache cleared ────────────────────────────────────────
  fresh)
    echo "Flushing Vite cache..."
    rm -rf "$ROOT/node_modules/.vite"
    echo "Serving Ch$CH (cache cleared) → http://localhost:3030"
    echo "Press O to open, S for speaker notes."
    npx --prefix "$ROOT" slidev "current.md" --force
    ;;

  # ── Dev server with hot-reload ────────────────────────────────────────────
  serve|*)
    echo "Serving Ch$CH → http://localhost:3030"
    echo "Press O to open, S for speaker notes."
    npx --prefix "$ROOT" slidev "current.md"
    ;;

esac
