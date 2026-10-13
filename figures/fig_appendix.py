"""Appendix figure (not in the Letter): upper tail of the
exact null of the comb test (CT), d_r = S/L, against its moment-matched gamma approximation
(run-length null, N = 100), log scale, so the region that decides a test is visible.
Recomputed here from code/dr_exact.py (seconds).  The gamma should track the exact tail
and, where it deviates, lie above it (conservative)."""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import C, COLUMN, RESULTS, ROOT, read_block, save

sys.path.insert(0, str(ROOT / "code"))
from dr_exact import calculate_exact_dr_distribution  # noqa: E402

Q = (16, 8, 4, 2, 1, 1)
N = 100
ALPHA = 0.05


def exact_tail():
    dist = calculate_exact_dr_distribution(N, list(Q))
    L = sum(Q)
    values = np.array([v for v, c in dist if c], dtype=float) / L
    counts = np.array([c for v, c in dist if c], dtype=float)
    p = counts / counts.sum()
    mean = (values * p).sum()
    var = ((values - mean) ** 2 * p).sum()
    shape, scale = mean ** 2 / var, var / mean
    survival = 1.0 - np.cumsum(p) + p          # P(T >= t)
    crit = values[np.searchsorted(np.cumsum(p), 1 - ALPHA)]
    return values, survival, shape, scale, crit


def main():
    fig, a = plt.subplots(figsize=(COLUMN, 1.1))

    values, survival, shape, scale, crit = exact_tail()
    keep = survival >= 1e-4
    lo = values[np.searchsorted(1 - survival, 0.5)]        # start at the median
    hi = values[keep].max()
    grid = np.linspace(lo, hi, 400)
    a.step(values, survival, where="post", color=C["dr"], label="exact")
    a.plot(grid, stats.gamma.sf(grid, shape, scale=scale), color=C["du"],
           linestyle="--", label="gamma")
    a.axvline(crit, color=C["grey"], linewidth=0.6, linestyle=":")
    a.axhline(ALPHA, color=C["grey"], linewidth=0.6, linestyle=":")
    a.set_yscale("log")
    a.set_xlim(lo, hi)
    a.set_ylim(1e-4, 1.0)
    a.set_xlabel(r"$t$   ($N{=}100$, run-length null)")
    a.set_ylabel(r"$P(d_{\mathbf{r}}\geq t)$")
    a.grid(True, which="both")
    a.legend(loc="upper right", frameon=False, handlelength=1.6)

    fig.tight_layout()
    save(fig, "fig_appendix.pdf")


if __name__ == "__main__":
    main()
