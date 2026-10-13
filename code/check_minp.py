"""Two checks of the min-p combination that the letter quotes (Section IV-B).

--part sidak   Why the minimum is calibrated by Monte Carlo instead of a
               Bonferroni or Sidak cut: the p-values of T(0), T(0.5), T(1) are
               strongly correlated, so the union-bound cuts are conservative
               (size about 0.02 at alpha = 0.05) and lose power.  Null batch A
               defines the marginal p-value transforms, batch B calibrates the
               minimum's critical value, batch C (independent) measures size, and
               the alternative draws measure power; corr(p0, p1) is on batch C.
--part phase   The combination inherits the phase-blindness of its members: on
               the geometric nulls of Fig. 1a its power is the same for both comb
               signs, while the uncentered d_u's (column DTV) is not.

Same nulls, alternatives, draws and statistics as experiment_selection_rule.py
and experiment_degeneracy.py (imported from them), so the numbers are on the
same footing as Table II and Fig. 1.
"""
import argparse

import numpy as np
from scipy import stats

from experiment_degeneracy import comb, geometric_null
from experiment_selection_rule import (EXPONENTS, NULLS, comb_alternative, combined,
                                       family, survival_transform)


def minp_machinery(rng, p0, N, null_draws):
    """Transforms from batch A and the minimum's alpha-quantile function from batch B."""
    a = family(rng.multinomial(N, p0, size=null_draws).astype(float), p0, N, EXPONENTS)
    transforms = {k: survival_transform(a[k]) for k in a if k.startswith("T(")}
    b = family(rng.multinomial(N, p0, size=null_draws).astype(float), p0, N, EXPONENTS)
    return transforms, combined(b, transforms, EXPONENTS)


def part_sidak(args):
    rng = np.random.default_rng(args.seed)
    k = len(EXPONENTS)
    sidak, bonf = 1.0 - (1.0 - args.alpha) ** (1.0 / k), args.alpha / k
    print(f"N={args.N}, alpha={args.alpha}; Sidak cut {sidak:.5f}, Bonferroni {bonf:.5f}, "
          f"MC cut per null below; {args.null_draws} null draws per batch, {args.alt_draws} alternative draws")
    print(f"{'null':>26} {'dep':>14} {'MC cut':>8} | {'size MC':>8} {'size Sid':>8} {'size Bon':>8} "
          f"| {'pow MC':>7} {'pow Sid':>7} {'pow Bon':>7} {'corr(p0,p1)':>11}")
    for name, weights in NULLS.items():
        p0 = weights / weights.sum()
        transforms, minp_b = minp_machinery(rng, p0, args.N, args.null_draws)
        cut = float(np.quantile(minp_b, args.alpha))
        c = family(rng.multinomial(args.N, p0, size=args.null_draws).astype(float), p0, args.N, EXPONENTS)
        minp_c = combined(c, transforms, EXPONENTS)
        p0v, p1v = transforms["T(0)"](c["T(0)"]), transforms["T(1)"](c["T(1)"])
        corr = float(np.corrcoef(p0v, p1v)[0, 1])
        sizes = [float(np.mean(minp_c <= t)) for t in (cut, sidak, bonf)]
        for kind in ("additive", "multiplicative"):
            p1 = comb_alternative(p0, kind, args.distance)
            alt = family(rng.multinomial(args.N, p1, size=args.alt_draws).astype(float), p0, args.N, EXPONENTS)
            minp_alt = combined(alt, transforms, EXPONENTS)
            powers = [float(np.mean(minp_alt <= t)) for t in (cut, sidak, bonf)]
            print(f"{name:>26} {kind:>14} {cut:>8.4f} | " + " ".join(f"{v:>8.3f}" for v in sizes)
                  + " | " + " ".join(f"{v:>7.3f}" for v in powers) + f" {corr:>11.3f}")


def part_phase(args):
    rng = np.random.default_rng(args.seed)
    print(f"min-p phase check: null p ∝ exp(-beta i), n={args.n}, N={args.N_phase}, "
          f"l1={args.distance_phase}, alpha={args.alpha}; {args.null_draws} null draws per batch, "
          f"{args.alt_draws} alternative draws per sign")
    print(f"{'beta':>5} {'minp +':>8} {'minp -':>8} {'ratio':>7} {'DTV +':>7} {'DTV -':>7} {'ratio':>9}")
    for beta in args.beta:
        p0 = geometric_null(args.n, beta)
        N = args.N_phase
        transforms, minp_b = minp_machinery(rng, p0, N, args.null_draws)
        cut = float(np.quantile(minp_b, args.alpha))
        null = family(rng.multinomial(N, p0, size=args.null_draws).astype(float), p0, N, EXPONENTS)
        dtv_cut = float(np.quantile(null["DTV"], 1.0 - args.alpha))
        out = {}
        for phase in (+1, -1):
            p1 = comb(p0, phase, args.distance_phase)
            alt = family(rng.multinomial(N, p1, size=args.alt_draws).astype(float), p0, N, EXPONENTS)
            out[phase] = (float(np.mean(combined(alt, transforms, EXPONENTS) <= cut)),
                          float(np.mean(alt["DTV"] > dtv_cut)))
        mp, mm = out[1][0], out[-1][0]
        dp, dm = out[1][1], out[-1][1]
        print(f"{beta:>5.2f} {mp:>8.3f} {mm:>8.3f} {max(mp, mm) / max(min(mp, mm), 1e-9):>7.2f} "
              f"{dp:>7.3f} {dm:>7.3f} {max(dp, dm) / max(min(dp, dm), 1e-9):>9.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=("sidak", "phase"), required=True)
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--distance", type=float, default=0.10)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--N_phase", type=int, default=1000)
    parser.add_argument("--distance_phase", type=float, default=0.08)
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.7, 1.0])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=200000)
    parser.add_argument("--alt_draws", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    (part_sidak if args.part == "sidak" else part_phase)(args)


if __name__ == "__main__":
    main()
