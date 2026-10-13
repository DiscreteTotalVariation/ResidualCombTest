"""l1-matched power over a catalogue of departure shapes, for non-uniform nulls.

This is the non-uniform counterpart of Table I of the parent paper.  Every
alternative is placed at the same l1 distance from the null, so the rows are
directly comparable, and every critical value is Monte-Carlo calibrated, so the
columns are compared at the same actual size.

Each shape is applied in the two ways a physical departure can arise:
  additive        p = p0 + c * s          the perturbation has constant amplitude
  multiplicative  p = p0 * (1 + c * s)    the perturbation follows the baseline
Under a flat null these coincide; under a sloped null they are different
alternatives, and the experiments show they call for different statistics.

Competitors include the order-aware baselines: AC is the alternating contrast,
the locally most powerful oracle for a known period-2 comb; vN is the von
Neumann ratio; RUN counts sign alternations of the centered counts.
"""

import argparse

import numpy as np

NULLS = {
    "uniform (n=8)": np.ones(8),
    "tent q=1:2:3:4:3:2:1": np.array([1., 2., 3., 4., 3., 2., 1.]),
    "ramp q=1:2:3:4:5:6": np.array([1., 2., 3., 4., 5., 6.]),
    "run-length q=16:8:4:2:1:1": np.array([16., 8., 4., 2., 1., 1.]),
}
KEYS = ["DTV", "DTV2", "CT", "T(1)", "min-p", "X2", "G", "CvM", "AD", "AC", "vN",
        "RUN"]
FAMILY = ("CT", "T(0.5)", "T(1)")


def survival_transform(reference):
    ordered = np.sort(reference)

    def transform(values):
        return 1.0 - np.searchsorted(ordered, values, side="left") / ordered.size
    return transform


def min_p(stats, transforms):
    return np.min([transforms[k](stats[k]) for k in FAMILY], axis=0)


def shapes(n):
    index = np.arange(n)
    return {
        "comb (period 2)": (-1.0) ** index,
        "comb (period 2, mirrored)": -(-1.0) ** index,
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


def build_alternative(p0, shape, kind, distance):
    """p0 perturbed along `shape`, kept a probability vector, at l1 `distance`."""
    delta = p0 * shape if kind == "multiplicative" else np.asarray(shape, float)
    delta = delta - delta.sum() * p0
    scale = np.abs(delta).sum()
    if scale == 0.0:
        return None
    delta = delta * distance / scale
    if (p0 + delta).min() <= 0.0:
        headroom = np.min(p0[delta < 0] / -delta[delta < 0])
        delta = delta * headroom * 0.999
    return p0 + delta


def statistics(x, p0, N):
    expected = N * p0
    centered = x - expected
    z = centered / np.sqrt(N * p0 * (1.0 - p0))
    alternating = np.where(np.arange(x.shape[1]) % 2 == 0, 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / expected), 0.0)
    # sign runs of the centered counts, ignoring exact zeros (a zero continues the
    # previous sign); the earlier version counted (+,0,+) as one change.
    centered_signs = np.sign(centered)
    index = np.where(centered_signs != 0, np.arange(centered_signs.shape[1])[None, :], -1)
    last = np.maximum.accumulate(index, axis=1)
    filled = np.where(last >= 0, np.take_along_axis(centered_signs, np.maximum(last, 0), axis=1), 0)
    changes = (filled[:, 1:] != filled[:, :-1]) & (filled[:, 1:] != 0) & (filled[:, :-1] != 0)
    uncentered = np.abs(np.diff(x, axis=1)).sum(1)
    # Cramer-von Mises and Anderson-Darling for ordered categories (Choulakian,
    # Lockhart & Stephens).  These are the natural order-aware competitors a
    # referee will reach for, and unlike the DTV family they accumulate the
    # departure along the CDF, so they should be strong against tilts and shifts
    # and weak against combs.  Including them is what makes the comparison fair.
    cdf_gap = (np.cumsum(x, axis=1) / N - np.cumsum(p0))[:, :-1]
    base = np.cumsum(p0)[:-1]
    # Choulakian, Lockhart and Stephens (1994), eqs. (1) and (3), weight each
    # squared CDF gap by p_j, NOT by a midpoint of adjacent cells.  An earlier
    # version used the midpoint; the difference is under 0.01 everywhere, but the
    # table cites that paper, so it must compute what that paper defines.
    weight = p0[:-1]
    return {
        "DTV": uncentered,
        "DTV2": uncentered,
        "CT": np.abs(np.diff(centered, axis=1)).sum(1),
        "T(1)": np.abs(np.diff(z, axis=1)).sum(1),
        "T(0.5)": np.abs(np.diff(centered / np.sqrt(N * p0 * (1.0 - p0)) ** 0.5, axis=1)).sum(1),
        "X2": (centered ** 2 / expected).sum(1),
        "G": 2.0 * terms.sum(1),
        "CvM": N * (cdf_gap ** 2 * weight).sum(1),
        "AD": N * (cdf_gap ** 2 / (base * (1.0 - base)) * weight).sum(1),
        "AC": np.abs(centered @ alternating),
        "vN": (np.diff(centered, axis=1) ** 2).sum(1) / (centered ** 2).sum(1),
        "RUN": changes.sum(1).astype(float),
    }


def rejection(null_stats, alt_stats, alpha):
    out = {}
    for key in null_stats:
        if key == "DTV2":
            low, high = np.quantile(null_stats[key], [alpha / 2, 1 - alpha / 2])
            out[key] = float(np.mean((alt_stats[key] > high)
                                     | (alt_stats[key] < low)))
        else:
            cut = np.quantile(null_stats[key], 1.0 - alpha)
            out[key] = float(np.mean(alt_stats[key] > cut))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--distance", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=200000)
    parser.add_argument("--alt_draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    N, alpha = args.N, args.alpha
    print(f"N={N}, l1 distance={args.distance}, alpha={alpha}, "
          "Monte-Carlo-calibrated critical values\n")

    for name, weights in NULLS.items():
        p0 = weights / weights.sum()
        n = len(p0)
        x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
        s0 = statistics(x0, p0, N)
        # min-p over the family T(t), t in {0, 0.5, 1}: marginal p-values from the
        # first null batch, the minimum's cut from an INDEPENDENT second batch
        # drawn from a separate generator so that every other column's Monte Carlo
        # stream (and hence every previously published number) is unchanged.
        rng_b = np.random.default_rng(args.seed + 1)
        s0b = statistics(rng_b.multinomial(N, p0, size=args.null_draws).astype(float), p0, N)
        transforms = {k: survival_transform(s0[k]) for k in FAMILY}
        cut_minp = np.quantile(min_p(s0b, transforms), alpha)
        for kind in ("additive", "multiplicative"):
            print(f"{name}   [{kind}]")
            print(f"{'shape':>18} " + " ".join(f"{k:>7}" for k in KEYS))
            for label, shape in shapes(n).items():
                p1 = build_alternative(p0, shape, kind, args.distance)
                if p1 is None:
                    continue
                x1 = rng.multinomial(N, p1, size=args.alt_draws).astype(float)
                s1 = statistics(x1, p0, N)
                r = rejection(s0, s1, alpha)
                r["min-p"] = float(np.mean(min_p(s1, transforms) < cut_minp))
                r.pop("T(0.5)", None)
                best = max(r, key=lambda k: r[k])
                row = " ".join(
                    (f"{r[k]:6.3f}*" if k == best else f"{r[k]:7.3f}")
                    for k in KEYS)
                print(f"{label:>18} {row}")
            print()


if __name__ == "__main__":
    main()
