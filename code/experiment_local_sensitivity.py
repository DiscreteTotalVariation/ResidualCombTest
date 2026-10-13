"""Directions of maximum and minimum sensitivity, in the local asymptotic limit.

A reviewer of the parent paper asked for the direction of maximum and minimum
sensitivity to be stated explicitly rather than inferred from a table of
hand-picked shapes.  This computes it.

Take a local alternative p_N = p0 + d / sqrt(N) with sum d_i = 0.  Then
    e / sqrt(N) = (x - N p0) / sqrt(N)  ->  d + u,     u ~ N(0, Sigma),
    Sigma = diag(p0) - p0 p0^T,
so every statistic has an N-free limit (e = centered counts; CT = d_r = T(0)):

    d_r / sqrt(N)    ->  sum_i |(d_i + u_i) - (d_{i-1} + u_{i-1})|
    T(1) / sqrt(N)   ->  sum_i |w_i - w_{i-1}|,  w_i = (d_i+u_i)/sqrt(p_i(1-p_i))
    chi2             ->  sum_i (d_i + u_i)^2 / p_i,  noncentral chi^2_{n-1}
                         with noncentrality  kappa(d) = sum_i d_i^2 / p_i.

The uncentered DTV d_u has no such limit, and that is the whole point.  Writing
d_u = sum_i |N (p_i - p_{i-1}) + sqrt(N) ((d_i+u_i) - (d_{i-1}+u_{i-1}))|, the
sign of every term with p_i != p_{i-1} is locked by the baseline slope once N is
large, so
    (d_u - N sum_i |p_i - p_{i-1}|) / sqrt(N)  ->
        sum_{i: dp_i != 0} sign(dp_i) * D_i  +  sum_{i: dp_i = 0} |D_i|,
    D_i = (d_i + u_i) - (d_{i-1} + u_{i-1}).
The first sum is LINEAR and telescopes.  This is the degeneracy, written as a
limit theorem rather than as a simulation artefact, and it is computed here
alongside the others.

The comparison is normalised so that Pearson's chi-square has the SAME local
power for every direction considered: kappa(d) = 1.  Chi-square is isotropic in
that metric, so it has no direction of maximum or minimum sensitivity at all.
Any spread in the other columns is therefore pure directional advantage, and the
best and worst directions are found by search rather than assumed.
"""

import argparse

import numpy as np
from scipy import stats

NULLS = {
    "uniform (n=8)": np.ones(8),
    "TPDF dither q=1:2:3:4:3:2:1": np.array([1., 2., 3., 4., 3., 2., 1.]),
    "ramp q=1:2:3:4:5:6": np.array([1., 2., 3., 4., 5., 6.]),
    "run-length q=16:8:4:2:1:1": np.array([16., 8., 4., 2., 1., 1.]),
}


def catalogue(n):
    """Structured departure directions, before normalisation."""
    index = np.arange(n)
    shapes = {
        "comb (period 2)": (-1.0) ** index,
        "period 3": np.cos(2 * np.pi * index / 3.0),
        "period 4": np.cos(2 * np.pi * index / 4.0),
        "block (++--)": (-1.0) ** (index // 2),
        "tilt (linear)": index - index.mean(),
        "half shift": np.where(index < n // 2, -1.0, 1.0),
        "single spike": np.where(index == n // 2, 1.0, 0.0),
        "edge spike": np.where(index == 0, 1.0, 0.0),
        "smooth bump": np.exp(-0.5 * ((index - (n - 1) / 2.0) / (n / 6.0)) ** 2),
        "half sine": np.sin(np.pi * (index + 0.5) / n),
    }
    return shapes


def normalise(d, p0):
    """Centre to sum zero and scale to unit chi-square noncentrality."""
    d = np.asarray(d, dtype=float) - np.mean(d)
    scale = np.sqrt(np.sum(d ** 2 / p0))
    if scale == 0.0:
        return None
    return d / scale


def limits(directions, noise, p0):
    """Limiting statistic values, shape (len(directions), len(noise))."""
    field = directions[:, None, :] + noise[None, :, :]
    differences = np.diff(field, axis=2)
    slope = np.diff(p0)
    locked = np.sign(slope)
    flat = slope == 0.0
    degenerate = (differences * locked).sum(2) + (np.abs(differences) * flat).sum(2)
    standardised = np.diff(field / np.sqrt(p0 * (1.0 - p0)), axis=2)
    return {
        "CT": np.abs(differences).sum(2),
        "T(1)": np.abs(standardised).sum(2),
        "DTVlim": degenerate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--search", type=int, default=4000,
                        help="random directions per null in the search stage")
    parser.add_argument("--search_draws", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    alpha = args.alpha

    for name, weights in NULLS.items():
        p0 = weights / weights.sum()
        n = len(p0)
        covariance = np.diag(p0) - np.outer(p0, p0)
        chol = np.linalg.cholesky(covariance + 1e-14 * np.eye(n))

        big = rng.standard_normal((args.draws, n)) @ chol.T
        small = big[:args.search_draws]

        zero = np.zeros((1, n))
        null_big = limits(zero, big, p0)
        thresholds = {k: np.quantile(v[0], 1.0 - alpha) for k, v in null_big.items()}
        # DTVlim (the limit of d_u) is used one-sided upper, as the CT is; its two-sided version is also
        # reported because that is the only way the degenerate statistic survives.
        low, high = np.quantile(null_big["DTVlim"][0], [alpha / 2, 1 - alpha / 2])
        null_small = limits(zero, small, p0)
        thresholds_small = {k: np.quantile(v[0], 1.0 - alpha)
                            for k, v in null_small.items()}

        chi_power = float(1.0 - stats.ncx2.cdf(
            stats.chi2.ppf(1.0 - alpha, df=n - 1), df=n - 1, nc=1.0))

        def evaluate(directions, noise, cuts):
            values = limits(directions, noise, p0)
            out = {k: (values[k] > cuts[k]).mean(1) for k in values}
            out["DTV2lim"] = ((values["DTVlim"] > high)
                              | (values["DTVlim"] < low)).mean(1)
            return out

        print("=" * 78)
        print(f"{name}   n={n}   "
              f"chi-square local power (identical for all directions) = "
              f"{chi_power:.3f}")
        print("=" * 78)
        header = (f"{'direction':>22} {'CT':>8} {'T(1)':>8} "
                  f"{'DTVlim':>8} {'DTV2lim':>8}")
        print(header)

        named, labels = [], []
        for label, shape in catalogue(n).items():
            d = normalise(shape, p0)
            if d is not None:
                named.append(d)
                labels.append(label)
        named = np.array(named)
        scores = evaluate(named, big, thresholds)
        for i, label in enumerate(labels):
            print(f"{label:>22} {scores['CT'][i]:8.3f} {scores['T(1)'][i]:8.3f} "
                  f"{scores['DTVlim'][i]:8.3f} {scores['DTV2lim'][i]:8.3f}")

        # Search for the extreme directions of the CT.
        candidates = rng.standard_normal((args.search, n))
        candidates -= candidates.mean(1, keepdims=True)
        candidates /= np.sqrt((candidates ** 2 / p0).sum(1))[:, None]
        rough = np.empty(args.search)
        for start in range(0, args.search, 200):
            block = candidates[start:start + 200]
            rough[start:start + 200] = evaluate(block, small,
                                                thresholds_small)["CT"]
        best = candidates[int(np.argmax(rough))]
        worst = candidates[int(np.argmin(rough))]
        for tag, seed_direction, sign in (("best found", best, 1.0),
                                          ("worst found", worst, -1.0)):
            current = seed_direction
            current_score = sign * evaluate(current[None, :], small,
                                            thresholds_small)["CT"][0]
            for step in (0.4, 0.2, 0.1, 0.05):
                for _ in range(30):
                    trial = current + step * rng.standard_normal((60, n))
                    trial -= trial.mean(1, keepdims=True)
                    trial /= np.sqrt((trial ** 2 / p0).sum(1))[:, None]
                    values = sign * evaluate(trial, small,
                                             thresholds_small)["CT"]
                    if values.max() > current_score:
                        current_score = values.max()
                        current = trial[int(np.argmax(values))]
            final = evaluate(current[None, :], big, thresholds)
            print(f"{tag:>22} {final['CT'][0]:8.3f} {final['T(1)'][0]:8.3f} "
                  f"{final['DTVlim'][0]:8.3f} {final['DTV2lim'][0]:8.3f}")
            correlation = float(np.corrcoef(current, normalise((-1.0) ** np.arange(n),
                                                               p0))[0, 1])
            print(f"{'':>22} shape {np.array2string(current / np.abs(current).max(), precision=2)}"
                  f"  corr with comb {correlation:+.3f}")
        print()


if __name__ == "__main__":
    main()
