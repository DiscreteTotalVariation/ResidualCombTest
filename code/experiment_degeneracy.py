"""Why the uncentered DTV d_u loses phase-blindness on a sloped null, measured
rather than asserted.  (DTV below means the uncentered d_u; CT means the comb
test on the centered DTV d_r.)

The parent paper's comb test detects a period-2 departure WITHOUT knowing its
phase.  That phase-blindness is its advertised virtue and the reason it is worth
having next to chi-square.

Under a sloped null the uncentered DTV d_u loses it.  Write d_i = p_i - p_{i-1} and
D_i = x_i - x_{i-1}, so E[D_i] = N d_i and Var[D_i] = N(p_i + p_{i-1} - d_i^2).
Define the locking margin

    rho = min_i  N |d_i| / sqrt( N (p_i + p_{i-1} - d_i^2) ) .

When rho is large every sign(D_i) equals sign(d_i) with probability tending to
one, and on that event

    DTV(x) = sum_i sign(d_i) D_i ,

a FIXED LINEAR functional of x.  For a monotone null every sign(d_i) is the same
and the sum telescopes to |x_n - x_1|: two cells, whatever happens in between.

A linear functional cannot be phase-blind.  A comb and its phase-reversed twin
move it in opposite directions, so one is detected and the other is
anti-detected.  This script measures all three consequences at once:

  telescoping   corr(DTV, x_1 - x_n) under the null, which should go to 1,
  linearity     fraction of draws on which the signs are fully locked,
  phase         power against the two phases of the same comb at the same l1.

The phase ratio max/min is the headline: for a genuine comb test it is 1.

The two phases are p0 + delta and p0 - delta with the SAME delta, so they are at
exactly the same l1 distance and the comparison is not confounded by one
alternative simply lying further away.  Note that they are still two DIFFERENT
alternatives, not related by any symmetry of the null, so even a phase-blind
statistic need not have a ratio of exactly 1: d_r is even in the centered counts, so
its two powers agree to leading order, but at finite N and finite l1 a small gap
remains.  The CT ratio columns are printed precisely so that this is checked
rather than assumed; the claim is that it stays O(1) while the DTV ratio does
not.
"""

import argparse

import numpy as np


def geometric_null(n, beta):
    weights = np.exp(-beta * np.arange(n))
    return weights / weights.sum()


def locking_margin(p0, N):
    d = np.diff(p0)
    variance = N * (p0[1:] + p0[:-1] - d ** 2)
    return float(np.min(N * np.abs(d) / np.sqrt(variance)))


def comb(p0, phase, distance):
    signs = phase * (-1.0) ** np.arange(len(p0))
    delta = p0 * signs
    delta = delta - delta.sum() * p0
    delta = delta * distance / np.abs(delta).sum()
    if (p0 + delta).min() <= 0.0:
        headroom = np.min(p0[delta < 0] / -delta[delta < 0])
        delta = delta * headroom * 0.999
    return p0 + delta


def statistics(x, p0, N):
    centered = x - N * p0
    return {"DTV": np.abs(np.diff(x, axis=1)).sum(1),
            "CT": np.abs(np.diff(centered, axis=1)).sum(1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--beta", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.4, 0.7, 1.0])
    parser.add_argument("--N", type=int, nargs="+", default=[50, 200, 1000])
    parser.add_argument("--distance", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=15)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"null p ∝ exp(-beta*i), n={args.n}; comb at l1={args.distance}, "
          f"alpha={args.alpha}, {args.draws} draws")
    print("rho = locking margin.  'locked' = fraction of null draws with every "
          "sign(x_i - x_{i-1}) = sign(p_i - p_{i-1}).")
    print("'corr' = corr(DTV, x_1 - x_n) under the null: 1 means DTV has "
          "collapsed to the endpoint contrast.")
    print("w* columns are power against the WORSE of the two phases, which is "
          "what a phase-blind test must deliver.\n")

    for N in args.N:
        print(f"N = {N}")
        print(f"{'beta':>5} {'rho':>6} {'locked':>7} {'corr':>6} | "
              f"{'DTV +':>7} {'DTV -':>7} {'ratio':>8} | "
              f"{'CT +':>7} {'CT -':>7} {'ratio':>6} | "
              f"{'wDTV':>7} {'wDTV2':>7} {'wCT':>7}")
        for beta in args.beta:
            p0 = geometric_null(args.n, beta)
            rho = locking_margin(p0, N)

            x0 = rng.multinomial(N, p0, size=args.draws).astype(float)
            s0 = statistics(x0, p0, N)
            differences = np.diff(x0, axis=1)
            slope_sign = np.sign(np.diff(p0))
            locked = float(np.mean(np.all(
                np.sign(differences) == slope_sign, axis=1)))
            endpoint = x0[:, 0] - x0[:, -1]
            corr = (float(np.corrcoef(s0["DTV"], endpoint)[0, 1])
                    if s0["DTV"].std() > 0 else float("nan"))

            cuts = {k: np.quantile(s0[k], 1.0 - args.alpha) for k in s0}
            # Two-sided uncentered DTV, equal-tailed: the obvious objection to a
            # one-sided test on a statistic whose mean falls under the
            # alternative, so it has to be carried explicitly.
            low2, high2 = np.quantile(s0["DTV"],
                                      [args.alpha / 2, 1.0 - args.alpha / 2])
            power = {}
            for phase in (1.0, -1.0):
                x1 = rng.multinomial(N, comb(p0, phase, args.distance),
                                     size=args.draws).astype(float)
                s1 = statistics(x1, p0, N)
                power[phase] = {k: float(np.mean(s1[k] > cuts[k])) for k in s1}
                power[phase]["DTV2"] = float(np.mean((s1["DTV"] > high2)
                                                     | (s1["DTV"] < low2)))

            def worst(key):
                return min(power[1.0][key], power[-1.0][key])

            def ratio(key):
                values = [power[1.0][key], power[-1.0][key]]
                return max(values) / max(min(values), 1e-9)

            print(f"{beta:>5.2f} {rho:>6.2f} {locked:>7.3f} {corr:>6.3f} | "
                  f"{power[1.0]['DTV']:>7.3f} {power[-1.0]['DTV']:>7.3f} "
                  f"{ratio('DTV'):>8.1f} | "
                  f"{power[1.0]['CT']:>7.3f} {power[-1.0]['CT']:>7.3f} "
                  f"{ratio('CT'):>6.2f} | {worst('DTV'):>7.3f} "
                  f"{worst('DTV2'):>7.3f} {worst('CT'):>7.3f}")
        print()


if __name__ == "__main__":
    main()
