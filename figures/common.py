"""Shared style for the letter's figures: IEEE column width, embedded fonts,
colour-blind-safe palette, one marker per statistic so the legend survives
greyscale printing."""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
IMG = ROOT / "paper" / "img"
IMG.mkdir(parents=True, exist_ok=True)

COLUMN = 3.5          # inches, IEEE single column
DOUBLE = 7.16

plt.rcParams.update({
    "font.family": "serif",
    # the letter is set in Nimbus Roman (IEEEtran's Times); STIX is its math companion.
    # Both are embedded, so beta, epsilon, X^2 inside a figure match the caption below it.
    "font.serif": ["Nimbus Roman", "Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito palette (colour-blind safe)
C = {
    "dr": "#0072B2",       # blue: the CT (centered, d_r)
    "du": "#D55E00",       # vermillion: uncentered d_u
    "x2": "#009E73",       # green
    "minp": "#56B4E9",     # sky blue
    "nist": "#CC79A7",     # purple
    "nist2": "#E69F00",    # orange
    "grey": "#7F7F7F",
    "black": "#000000",
}

STYLE = {
    "dr":    dict(color=C["dr"],  marker="o", linestyle="-"),
    "du":    dict(color=C["du"],   marker="s", linestyle="-"),
    "min-p": dict(color=C["minp"], marker="D", linestyle="--"),
    "X2":    dict(color=C["x2"],   marker="^", linestyle="-"),
    "ser3":  dict(color=C["nist"], marker="v", linestyle="-."),
    "apen":  dict(color=C["nist2"],marker="P", linestyle="-."),
    "blind": dict(color=C["grey"], marker="x", linestyle=":"),
    # distinct marker AND dash pattern so it survives greyscale printing
    "lrun": dict(color=C["black"], marker="D", linestyle=(0, (4, 1, 1, 1)),
                 markersize=2.6, markerfacecolor="none"),
}


def read_block(path, start, stop=None):
    """Lines of `path` from the first line starting with `start` up to (not
    including) the first later line starting with `stop` (or EOF)."""
    lines = pathlib.Path(path).read_text().splitlines()
    out, active = [], False
    for line in lines:
        if not active and line.startswith(start):
            active = True
            continue
        if active:
            if stop is not None and line.startswith(stop):
                break
            out.append(line)
    return out


def save(fig, name):
    path = IMG / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
