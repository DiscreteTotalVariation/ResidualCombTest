"""Choosing between T(0) = d_r (the CT) and T(1) without knowing the alternative.

Three separate experiments (variants part A, the shape table, the run-length RNG)
agree that T(0) and T(1) split the alternatives between them:

  multiplicative departures, p = p0 * (1 + c*s)   ->  T(0) wins,
  additive departures,       p = p0 + c*s        ->  T(1) wins.

That is useful only if a practitioner can tell which case they are in, and in
general they cannot: the departure is what is being tested for.  So a selection
rule based on the alternative is circular.

The honest way out is not to select but to combine.  Both statistics are members
of one family indexed by an exponent t,

  T(t) = sum_i | y_i - y_{i-1} |,     y_i = e_i / (N p_i (1 - p_i))^(t/2),

with e_i = x_i - N p_i the centered counts, T(0) = d_r the CT statistic and
T(1) the DTV of the adjusted (standardized) residuals.  Each T(t) is converted
to a p-value under the null and the combined statistic is the minimum of those
p-values over a small grid of t.  The combination costs a multiplicity penalty but needs no knowledge of
the alternative.

The null calibration is split across two independent Monte-Carlo batches: the
first defines the marginal p-value transforms, the second calibrates the critical
value of the minimum.  Reusing one batch for both would bias the size downwards,
which is exactly the error the experiment is meant to rule out, so the reported
size column is a genuine check.
"""

import argparse

import numpy as np

NULLS = {
    "uniform (n=8)": np.ones(8),
    "tent q=1:2:3:4:3:2:1": np.array([1., 2., 3., 4., 3., 2., 1.]),
    "ramp q=1:2:3:4:5:6": np.array([1., 2., 3., 4., 5., 6.]),
    "run-length q=16:8:4:2:1:1": np.array([16., 8., 4., 2., 1., 1.]),
}
EXPONENTS = (0.0, 0.5, 1.0)


def family(x, p0, N, exponents):
    """The statistics T(t), plus chi-square and the uncentered DTV d_u for reference."""
    centered = x - N * p0
    scale = np.sqrt(N * p0 * (1.0 - p0))
    out = {}
    for t in exponents:
        y = centered / scale ** t
        out[f"T({t:g})"] = np.abs(np.diff(y, axis=1)).sum(1)
    out["X2"] = (centered ** 2 / (N * p0)).sum(1)
    out["DTV"] = np.abs(np.diff(x, axis=1)).sum(1)
    return out


def survival_transform(reference):
    """p-value of each entry of `values` under the empirical law of `reference`.

    Returns a function; larger statistic -> smaller p-value.  Ties are handled by
    counting reference points strictly below, which keeps the transform valid for
    the lattice-valued statistics.
    """
    ordered = np.sort(reference)
    size = ordered.size

    def transform(values):
        return 1.0 - np.searchsorted(ordered, values, side="left") / size

    return transform


def combined(stats, transforms, exponents):
    return np.min([transforms[f"T({t:g})"](stats[f"T({t:g})"])
                   for t in exponents], axis=0)


def comb_alternative(p0, kind, distance):
    signs = (-1.0) ** np.arange(len(p0))
    delta = p0 * signs if kind == "multiplicative" else signs
    delta = delta - delta.sum() * p0
    delta = delta * distance / np.abs(delta).sum()
    if (p0 + delta).min() <= 0.0:
        headroom = np.min(p0[delta < 0] / -delta[delta < 0])
        delta = delta * headroom * 0.999
    return p0 + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--distance", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=200000)
    parser.add_argument("--alt_draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--exponents", type=float, nargs="+", default=list(EXPONENTS),
                        help="t grid for the min-p combination (default 0 0.5 1)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    N, alpha = args.N, args.alpha
    keys = [f"T({t:g})" for t in tuple(args.exponents)] + ["min-p", "X2", "DTV"]

    print(f"N={N}, l1 distance={args.distance}, alpha={alpha}")
    print("T(0)=CT (d_r), T(1)=DTV of adjusted residuals, min-p = combination over t in "
          f"{{{', '.join(f'{t:g}' for t in tuple(args.exponents))}}}")
    print("null calibration split across two independent batches, so 'size' is a "
          "real check\n")

    print(f"{'null':>26} {'departure':>15} " + " ".join(f"{k:>8}" for k in keys))
    regret = {key: [] for key in keys}
    for name, weights in NULLS.items():
        p0 = weights / weights.sum()

        shape_batch = rng.multinomial(N, p0, size=args.null_draws)
        calib_batch = rng.multinomial(N, p0, size=args.null_draws)
        shape_stats = family(shape_batch.astype(float), p0, N, tuple(args.exponents))
        calib_stats = family(calib_batch.astype(float), p0, N, tuple(args.exponents))

        transforms = {key: survival_transform(shape_stats[key])
                      for key in shape_stats}
        cuts = {key: np.quantile(shape_stats[key], 1.0 - alpha)
                for key in shape_stats}
        min_p_cut = np.quantile(combined(calib_stats, transforms, tuple(args.exponents)),
                                alpha)

        for label, sample in [("size (null)", p0)] + [
                (kind, comb_alternative(p0, kind, args.distance))
                for kind in ("multiplicative", "additive")]:
            x = rng.multinomial(N, sample, size=args.alt_draws).astype(float)
            stats = family(x, p0, N, tuple(args.exponents))
            rates = {key: float(np.mean(stats[key] > cuts[key]))
                     for key in stats}
            rates["min-p"] = float(np.mean(
                combined(stats, transforms, tuple(args.exponents)) <= min_p_cut))
            best = max((k for k in keys if k != "min-p"), key=lambda k: rates[k])
            row = " ".join(
                (f"{rates[k]:7.3f}*" if k == best else f"{rates[k]:8.3f}")
                for k in keys)
            print(f"{name:>26} {label:>15} {row}")
            if label != "size (null)":
                for key in keys:
                    regret[key].append(rates[best] - rates[key])
        print()

    print("worst-case shortfall against the best fixed statistic in each cell")
    print("(the price of committing to one statistic before seeing the data)")
    print(f"{'statistic':>26} {'worst':>8} {'mean':>8}")
    for key in keys:
        values = np.array(regret[key])
        print(f"{key:>26} {values.max():>8.3f} {values.mean():>8.3f}")


if __name__ == "__main__":
    main()
