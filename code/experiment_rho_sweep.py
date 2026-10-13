"""Does the collapse index have a THRESHOLD, or does it order the loss smoothly?

experiment_rho_diagnostic.py evaluates ten nulls, and their collapse indices
happen to cluster below 0.51 and above 0.89 -- a gap in the SAMPLE, which an
earlier version of the letter mistook for a gap in the relationship and reported
as a threshold near 0.75.  This script fills the range: 41 nulls spanning the
whole index axis, same settings as the diagnostic (l1 = 0.08, alpha = 0.05,
40000 null draws, both comb signs, the WORSE sign reported).

The result is that the worse-sign power of the uncentered d_u (one-sided,
labelled DTV below) declines monotonically with the index and only falls to the
nominal level above about 0.9, while the comb test (CT, on the centered d_r) is
unaffected everywhere.  The index is therefore an ordering, not a cut-off: the
Spearman correlation of the index with d_u's worse-sign power is NEGATIVE.
"""

import argparse

import numpy as np
from scipy.stats import spearmanr


def comb(p0, phase, distance):
    """p0 perturbed along a period-2 comb of the given phase, at l1 `distance`."""
    signs = phase * (-1.0) ** np.arange(len(p0))
    delta = p0 * signs
    delta = delta - delta.sum() * p0
    delta = delta * distance / np.abs(delta).sum()
    if (p0 + delta).min() <= 0.0:
        headroom = np.min(p0[delta < 0] / -delta[delta < 0])
        delta = delta * headroom * 0.999
    return p0 + delta


def nulls(rng):
    out = []
    for beta in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        for n in (6, 10):
            w = np.exp(-beta * np.arange(n))
            out.append((f"geometric beta={beta} n={n}", w / w.sum()))
    for a in (0.2, 0.5, 1.0, 3.0, 10.0):
        for _ in range(4):
            n = int(rng.integers(5, 20))
            w = rng.dirichlet(np.ones(n) * a)
            if w.min() < 1e-4:
                continue
            out.append((f"Dirichlet a={a} n={n}", w))
    for n in (6, 8, 12):
        w = np.arange(1, n + 1.0)
        out.append((f"ramp n={n}", w / w.sum()))
        w = np.minimum(np.arange(1, n + 1.0), n - np.arange(n))
        out.append((f"tent n={n}", w / w.sum()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--distance", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=40000)
    parser.add_argument("--alt_draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=22)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"N={args.N}, comb at l1={args.distance}, alpha={args.alpha}, "
          f"{args.draws} null / {args.alt_draws} alternative draws per sign; "
          "worst = the smaller power over the two comb phases")
    print("index = collapse index = corr(sum|x_i - x_{i-1}|, sum sign(dp_i)(x_i - x_{i-1})) under H0\n")
    print("sweep rows")                       # machine-readable block for figures/fig1.py
    print(f"{'null':>24} {'index':>7} {'DTV worst':>10} {'CT worst':>10}")
    rows = []
    for name, p0 in nulls(rng):
        x0 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
        e0 = x0 - args.N * p0
        dtv0 = np.abs(np.diff(x0, axis=1)).sum(1)
        ct0 = np.abs(np.diff(e0, axis=1)).sum(1)
        linear0 = (np.sign(np.diff(p0)) * np.diff(x0, axis=1)).sum(1)
        index = float(np.corrcoef(dtv0, linear0)[0, 1]) if linear0.std() > 0 else 0.0
        cut_d = float(np.quantile(dtv0, 1.0 - args.alpha))
        cut_r = float(np.quantile(ct0, 1.0 - args.alpha))
        powers = []
        for phase in (+1, -1):
            x1 = rng.multinomial(args.N, comb(p0, phase, args.distance),
                                 size=args.alt_draws).astype(float)
            e1 = x1 - args.N * p0
            powers.append((float(np.mean(np.abs(np.diff(x1, axis=1)).sum(1) > cut_d)),
                           float(np.mean(np.abs(np.diff(e1, axis=1)).sum(1) > cut_r))))
        worst_d = min(p[0] for p in powers)
        worst_r = min(p[1] for p in powers)
        rows.append((index, worst_d, worst_r, name))
        print(f"{name:>24} {index:7.3f} {worst_d:10.3f} {worst_r:10.3f}")

    print("end sweep rows")
    rows.sort()
    print(f"\n{len(rows)} nulls, sorted by index; monotonicity of the uncentered d_u's loss:")
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 0.97), (0.97, 1.01)):
        band = [r for r in rows if lo <= r[0] < hi]
        if not band:
            continue
        print(f"  index [{lo:.2f}, {hi:.2f}): {len(band):2d} nulls, "
              f"DTV worst {min(r[1] for r in band):.3f} to {max(r[1] for r in band):.3f}, "
              f"CT worst {min(r[2] for r in band):.3f} to {max(r[2] for r in band):.3f}")
    keeps = [r for r in rows if r[1] > 1.5 * args.alpha]
    lost = [r for r in rows if r[1] <= 1.5 * args.alpha]
    print(f"  d_u still detects the worse sign: index up to {max(r[0] for r in keeps):.3f}")
    print(f"  d_u has lost the worse sign:      index from {min(r[0] for r in lost):.3f}")
    # Spearman of the index with d_u's worse-sign power itself (the Letter's sign:
    # index up, power down, so negative), tied powers given their average rank.
    # Until 2026-10-13 this correlated argsort-ranks with MINUS the power, which
    # printed a positive value and broke the many ties (equal powers) arbitrarily.
    spearman = float(spearmanr([r[0] for r in rows], [r[1] for r in rows]).correlation)
    print(f"  so no clean cut exists below {min(r[0] for r in lost):.2f}; the index ORDERS the loss "
          f"(Spearman {spearman:.3f})")
    print(f"  CT worse-sign power over all {len(rows)} nulls: "
          f"{min(r[2] for r in rows):.3f} to {max(r[2] for r in rows):.3f}")


if __name__ == "__main__":
    main()
