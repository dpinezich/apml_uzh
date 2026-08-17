"""Shared style, colours and save helpers for slide images and GIF animations.

Usage inside imagegen/chNN.py::

    from imagegen.common import *
    def generate():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ...
        save(fig, "my_image.png", ["ch03"])
        save_gif(fig, update_fn, n_frames, "my_anim.gif", ["ch03"], fps=2)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

__all__ = [
    "np", "plt", "mpatches", "ROOT", "PUBLIC", "CHAPTERS",
    "TEAL", "TEAL_DARK", "DARK", "MUTED", "BORDER", "BG", "RED", "GREEN",
    "BLUE", "ORANGE", "PURPLE", "apml_style", "save", "save_gif",
]

ROOT   = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "slidev" / "public"
PUBLIC.mkdir(parents=True, exist_ok=True)

CHAPTERS = {
    "ch01": ROOT / "1-introduction" / "01-slides",
    "ch02": ROOT / "2-selection_cleaning_preparing" / "01-slides",
    "ch03": ROOT / "3-supervised_learning" / "01-slides",
    "ch04": ROOT / "3-supervised_learning" / "01-slides",
    "ch05": ROOT / "3-supervised_learning" / "01-slides",
    "ch06": ROOT / "3-supervised_learning" / "01-slides",
    "ch07": ROOT / "4-unsupervised_learning" / "01-slides",
    "ch08": ROOT / "4-unsupervised_learning" / "01-slides",
    "ch09": ROOT / "4-unsupervised_learning" / "01-slides",
    "ch10": ROOT / "5-reinforcement_learning" / "01-slides",
    "ch11": ROOT / "5-reinforcement_learning" / "01-slides",
    "ch12": ROOT / "6-capstone_ml" / "01-slides",
}

# ── Brand colours (same as generate_images.py) ─────────────────────────────
TEAL      = "#00CCCC"
TEAL_DARK = "#009090"
DARK      = "#1a1a1a"
MUTED     = "#666666"
BORDER    = "#e0e0e0"
BG        = "#ffffff"
RED       = "#e74c3c"
GREEN     = "#2ecc71"
BLUE      = "#3498db"
ORANGE    = "#e67e22"
PURPLE    = "#9b59b6"


def apml_style():
    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   DARK,
        "axes.titlecolor":   DARK,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.color":       MUTED,
        "ytick.color":       MUTED,
        "grid.color":        BORDER,
        "grid.linewidth":    0.8,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "savefig.dpi":       150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": BG,
    })


apml_style()


def _targets(name, chapters):
    dirs = {CHAPTERS[ch] for ch in chapters}
    return [d / name for d in dirs] + [PUBLIC / name]


def save(fig, name: str, chapters: list):
    """Save a static figure to every listed chapter's 01-slides/ AND slidev/public/."""
    for dest in _targets(name, chapters):
        fig.savefig(dest)
    plt.close(fig)
    print(f"  ✓  {name}")


def save_gif(fig, update, n_frames: int, name: str, chapters: list,
             fps: int = 2, dpi: int = 100, hold_last: int = 0):
    """Render a GIF via FuncAnimation and write it next to the PNGs.

    ``update(i)`` mutates the figure for frame i.  ``hold_last`` repeats the
    final frame that many extra times so the loop pauses on the result.
    Also writes ``<name>.png`` (final frame) as static fallback for PDF export
    — reference the GIF in slides and the PNG in speaker notes if needed.
    """
    frames = list(range(n_frames)) + [n_frames - 1] * hold_last
    anim = FuncAnimation(fig, lambda i: update(i), frames=frames, blit=False)
    targets = _targets(name, chapters)
    anim.save(targets[0], writer=PillowWriter(fps=fps), dpi=dpi)
    data = targets[0].read_bytes()
    for dest in targets[1:]:
        dest.write_bytes(data)
    # static fallback = last frame
    update(n_frames - 1)
    png = Path(name).with_suffix(".png").name
    for dest in _targets(png, chapters):
        fig.savefig(dest, dpi=dpi)
    plt.close(fig)
    print(f"  ✓  {name}  ({len(frames)} frames, +{png})")
