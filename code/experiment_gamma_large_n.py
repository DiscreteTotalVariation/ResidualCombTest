"""Gamma calibration of the CT statistic d_r far beyond the exact regime.

The exact dynamic program reaches N of a few hundred.  The applications run at
N = 500 to 5*10^4, so the paper has to show that the parent's gamma route
survives the move to a non-uniform null and to the centered DTV d_r.  The
parent verified its own gamma FPR up to N = 5*10^4; this is the same check for
d_r, and it is the check that licenses every applied number in the paper.

Method, deliberately identical to the parent's: moment-match a gamma to the null
statistic from K draws, read the critical value off the fitted gamma, then
measure the attained false positive rate on an INDEPENDENT batch of K draws.
Reusing one batch would flatter the fit.

Companion columns do the same for T(0.5) and T(1), because the
recommended procedure is the min-p combination over t in {0, 0.5, 1} and a
combination is only as well calibrated as its parts.  (T(0.5) was added on
2026-09-02 after a plan review noticed that the middle member of the family had
never been checked.)
"""

import argparse

import numpy as np
from scipy import stats

NULLS = {
    "uniform n=8": np.ones(8),
    "tent 1:2:3:4:3:2:1": np.array([1., 2., 3., 4., 3., 2., 1.]),
    "ramp 1:2:3:4:5:6": np.array([1., 2., 3., 4., 5., 6.]),
    "run-length 16:8:4:2:1:1": np.array([16., 8., 4., 2., 1., 1.]),
    "image-like n=64": np.exp(-0.5 * ((np.arange(64) - 31.5) / 12.0) ** 2) + 0.05,
}


def statistics(x, p0, N):
    centered = x - N * p0
    z = centered / np.sqrt(N * p0 * (1.0 - p0))
    half = centered / np.sqrt(np.sqrt(N * p0 * (1.0 - p0)))
    return {"CT": np.abs(np.diff(centered, axis=1)).sum(1),
            "T(0.5)": np.abs(np.diff(half, axis=1)).sum(1),
            "T(1)": np.abs(np.diff(z, axis=1)).sum(1)}


def gamma_fit(sample):
    """Moment-matched gamma, as in the parent paper."""
    mean, variance = sample.mean(), sample.var()
    shape = mean ** 2 / variance
    scale = variance / mean
    return shape, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+",
                        default=[500, 2000, 10000, 50000])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"moment-matched gamma, K={args.draws} to fit and K to test "
          f"(independent), alpha={args.alpha}")
    print("FPR should sit at alpha; KS is between the fitting batch and the "
          "fitted gamma\n")

    for name, weights in NULLS.items():
        p0 = weights / weights.sum()
        print(f"{name}   (n={len(p0)})")
        print(f"{'N':>7} {'CT FPR':>10} {'CT KS':>9} "
              f"{'T(.5) FPR':>10} {'T(.5) KS':>9} "
              f"{'T(1) FPR':>10} {'T(1) KS':>9}")
        for N in args.N:
            fit_batch = rng.multinomial(N, p0, size=args.draws).astype(float)
            test_batch = rng.multinomial(N, p0, size=args.draws).astype(float)
            fit_stats = statistics(fit_batch, p0, N)
            test_stats = statistics(test_batch, p0, N)
            row = []
            for key in ("CT", "T(0.5)", "T(1)"):
                shape, scale = gamma_fit(fit_stats[key])
                critical = stats.gamma.isf(args.alpha, shape, scale=scale)
                fpr = float(np.mean(test_stats[key] > critical))
                ks = float(stats.kstest(fit_stats[key],
                                        stats.gamma(shape, scale=scale).cdf)
                           .statistic)
                row.extend([fpr, ks])
            print(f"{N:>7} {row[0]:>10.4f} {row[1]:>9.4f} "
                  f"{row[2]:>10.4f} {row[3]:>9.4f} "
                  f"{row[4]:>10.4f} {row[5]:>9.4f}")
        print()


if __name__ == "__main__":
    main()
