"""The Letter's single figure (double column, three panels).

(a) the uncentered d_u loses phase-blindness as the null steepens while the
    comb test (CT, on centered counts) does not -- results/experiment_degeneracy.txt, N = 1000;
(b) the collapse index, computed from the null alone, orders that loss over 52 nulls
    -- results/experiment_rho_diagnostic.txt (N = 2000) and experiment_rho_sweep.txt;
(c) the two-lane bit source: CT against chi-square and six NIST tests on the same
    2000-bit windows -- results/experiment_runlength_rng.txt, Part B.

Drawn at the full text width (7.16 in) so nothing is scaled down on the page and
all type prints at 7-8 pt.  Higher power is better throughout; the dashed line is
the nominal size alpha = 0.05."""

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import C, DOUBLE, RESULTS, STYLE, read_block, save


# ----------------------------------------------------------------- data readers
def degeneracy(N=1000):
    rows = []
    for line in read_block(RESULTS / "experiment_degeneracy.txt", f"N = {N}", "N = "):
        parts = line.replace("|", " ").split()
        if not parts or not re.match(r"^\d", parts[0]):
            continue
        beta, rho, locked, corr, dp, dm, _r, tp, tm = map(float, parts[:9])
        rows.append((beta, dp, dm, tp, tm))
    return np.array(rows)


def sweep():
    """The 42-null sweep: (index, worse-sign DTV power, worse-sign CT power)."""
    pts = []
    for line in read_block(RESULTS / "experiment_rho_sweep.txt", "sweep rows", "end sweep rows"):
        parts = line.split()
        nums = [p for p in parts if re.match(r"^-?\d+\.\d+$", p)]
        if len(nums) >= 3:
            pts.append(tuple(float(v) for v in nums[-3:]))
    return pts


def diagnostic(N=2000):
    pts = []
    for line in read_block(RESULTS / "experiment_rho_diagnostic.txt", f"N = {N}", "N = "):
        if "|" not in line or "null" in line or "linearity <" in line:
            continue
        parts = line.replace("|", " ").split()
        nums = [float(x) for x in parts if re.match(r"^-?\d+(\.\d+)?$", x)]
        lin, dtv_worst, tres_worst = nums[-4], nums[-2], nums[-1]
        label = " ".join(p for p in parts if not re.match(r"^-?\d+(\.\d+)?$", p))
        pts.append((label, lin, dtv_worst, tres_worst))
    return pts


def part_b(width=2000):
    rows, cols = [], None
    for line in read_block(RESULTS / "experiment_runlength_rng.txt",
                           f"windows of {width} bits", "windows of "):
        parts = line.split()
        if parts and parts[0] == "eps":
            cols = parts
        elif parts and cols and re.match(r"^\d", parts[0]):
            rows.append([float(x) for x in parts[:len(cols)]])
    return dict(zip(cols, np.array(rows).T))


def best_of(d, keys):
    """Power of the best block length per eps: an oracle choice that favors NIST."""
    return np.max([d[k] for k in keys if k in d], axis=0)


# ----------------------------------------------------------------------- panels
def panel_a(a):
    beta, dp, dm, tp, tm = degeneracy().T
    a.plot(beta, tp, color=C["dr"], marker="o", zorder=4, label="CT, sign $+$")
    a.plot(beta, tm, color=C["dr"], marker="o", markerfacecolor="white", linestyle="--",
           zorder=4, label="CT, sign $-$")
    a.plot(beta, dp, color=C["du"], marker="s", zorder=3, label=r"$d_{\mathbf{u}}$, sign $+$")
    a.plot(beta, dm, color=C["du"], marker="s", markerfacecolor="white", linestyle="--",
           zorder=3, label=r"$d_{\mathbf{u}}$, sign $-$")
    a.set_xlabel(r"null slope $\beta$ in $r_i\propto \exp(-\beta i)$")
    a.set_ylabel("power")
    # legends sit in the band above power 1, which no curve reaches
    a.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.4,
             columnspacing=0.8, borderaxespad=0.15, handletextpad=0.4, labelspacing=0.25)


def panel_b(b):
    for lin, dtv_worst, tres_worst in sweep():
        b.plot(lin, tres_worst, marker=".", color=C["dr"], linestyle="none",
               markersize=4, alpha=0.65, zorder=2)
        b.plot(lin, dtv_worst, marker="x", color=C["du"], linestyle="none",
               markersize=3.4, markeredgewidth=0.8, alpha=0.65, zorder=2)
    for label, lin, dtv_worst, tres_worst in diagnostic():
        image = label.startswith("image")
        b.plot(lin, tres_worst, marker="D" if image else "o", color=C["dr"], linestyle="none",
               markersize=4.2 if image else 4.6, zorder=3)
        b.plot(lin, dtv_worst, marker="p" if image else "s", color=C["du"], linestyle="none",
               markersize=5 if image else 4.6, zorder=3)
    b.set_xlabel("collapse index of the null")
    b.set_ylabel("power, worse comb sign")
    b.set_xlim(-0.05, 1.05)
    handles = [
        Line2D([], [], color=C["dr"], marker="o", linestyle="none", markersize=4.6, label="CT"),
        Line2D([], [], color=C["du"], marker="s", linestyle="none", markersize=4.6, label=r"$d_{\mathbf{u}}$"),
        Line2D([], [], color=C["dr"], marker="D", linestyle="none", markersize=4.2, label="CT, image"),
        Line2D([], [], color=C["du"], marker="p", linestyle="none", markersize=5, label=r"$d_{\mathbf{u}}$, image"),
        Line2D([], [], color=C["dr"], marker=".", linestyle="none", markersize=4, label="CT, further"),
        Line2D([], [], color=C["du"], marker="x", linestyle="none", markersize=3.4, markeredgewidth=0.8, label=r"$d_{\mathbf{u}}$, further"),
    ]
    # two columns, CT left and d_u right: rows are (plain, image, further)
    handles = [handles[0], handles[2], handles[4], handles[1], handles[3], handles[5]]
    b.legend(handles=handles, loc="upper left", ncol=2, frameon=False, handlelength=1.0,
             columnspacing=1.0, borderaxespad=0.15, handletextpad=0.4, labelspacing=0.25)


def panel_c(c):
    d = part_b()
    eps = d["eps"]
    c.plot(eps, d["CT"], label="CT", zorder=5, **STYLE["dr"])
    c.plot(eps, d["X2"], label=r"$X^2$", zorder=4, **STYLE["X2"])
    c.plot(eps, d["lrun"], label="longest run", zorder=4, **STYLE["lrun"])
    c.plot(eps, best_of(d, [f"ser{m}" for m in (3, 4, 5, 6)] + [f"ser{m}d" for m in (3, 4, 5, 6)]),
           label="serial", **STYLE["ser3"])
    c.plot(eps, best_of(d, [f"apen{m}" for m in (2, 3, 4)]),
           label="approx. entropy", **STYLE["apen"])
    blind = np.max([d["mono"], d["runs"], d["ser2"], d["dft"]], axis=0)
    c.plot(eps, blind, label="4 others, max", **STYLE["blind"])
    c.plot(eps, d["DTV"], label=r"$d_{\mathbf{u}}$", **STYLE["du"])
    c.set_xlabel(r"lane mismatch $\varepsilon$")
    c.set_ylabel("power")
    c.set_xlim(-0.005, 0.155)
    c.set_xticks([0, 0.05, 0.10, 0.15])
    c.legend(loc="upper left", ncol=2, frameon=False, handlelength=1.6, borderaxespad=0.15,
             handletextpad=0.5, labelspacing=0.25, columnspacing=0.9)


def main():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8})
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DOUBLE, 1.9))
    for ax, tag in zip((a, b, c), "abc"):
        ax.axhline(0.05, color="black", linewidth=0.6, linestyle="--", zorder=1)
        ax.set_ylim(-0.04, 1.47)        # the band above 1 holds the legend
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(True)
        ax.set_title(f"({tag})", loc="left", pad=3)
    panel_a(a)
    panel_b(b)
    panel_c(c)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_main.pdf")


if __name__ == "__main__":
    main()
