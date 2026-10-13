"""Exact power, with no Monte Carlo anywhere.

A reviewer of the parent paper objected that comparing an exact test against
asymptotic competitors confounds power with size distortion.  Monte Carlo
calibration removes that confound but introduces sampling error.  Here both are
removed: the null distribution, the critical value and the power are all
computed exactly from the integer dynamic programs, so every number below is a
rational number rounded for printing, not an estimate.

For each null q0 and comb alternative q1 (both integer weight vectors) we report
  crit    the exact critical value, the smallest s with P_{q0}(S > s) <= alpha
  size    P_{q0}(S > crit), exactly, necessarily <= alpha
  power   P_{q1}(S > crit), exactly
for the comb test (CT) on the centered DTV, S = L*d_r, and the same quantities
for the uncentered DTV d_u of the parent paper (the DTV columns).

The point of the DTV (d_u) rows is PHASE DEPENDENCE.  The comb test of the parent
paper detects a period-2 departure WITHOUT knowing its phase, and that
phase-blindness is its advertised virtue.  Under a sloped null the uncentered
d_u loses it: once the adjacent-pair signs lock, the DTV collapses to a fixed linear
functional, and a linear functional cannot be phase-blind, so the two phases move
it in OPPOSITE directions.  One phase is detected and the other is
anti-detected, sometimes with power below size.  A statistic whose answer depends
on the phase of the departure is not a comb test.

Because that claim is a COMPARISON BETWEEN THE TWO PHASES, the two phases must be
equally hard, otherwise the asymmetry could be an artefact of one alternative
simply lying further from the null.  An earlier version of this script multiplied
the weights by 6 and 4 alternately, which does not renormalise symmetrically and
left the two phases up to 13% apart in l1 (0.1699 vs 0.1925 on the run-length
null).  `comb_pair` below removes that confound: it builds the two alternatives
as p0 +/- delta with the SAME delta, so their l1 distances are equal by
construction, exactly and not approximately.
"""

import argparse
from fractions import Fraction
from math import gcd

from du_exact import calculate_exact_dtv_distribution_nonuniform
from dr_exact import calculate_exact_dr_distribution

NULLS = [
    ("uniform control  q0=1:1:1:1:1", (1, 1, 1, 1, 1)),
    ("comb null        q0=1:2:1:2:1", (1, 2, 1, 2, 1)),
    ("ramp             q0=1:2:3:4:5", (1, 2, 3, 4, 5)),
    ("tent             q0=1:2:3:4:3:2:1", (1, 2, 3, 4, 3, 2, 1)),
    ("run-length       q0=16:8:4:2:1:1", (16, 8, 4, 2, 1, 1)),
]


def comb_pair(q0, distance):
    """The two phase-reversed multiplicative combs at EXACTLY equal l1 distance.

    `step` is the multiplicative comb q_i * (-1)^i projected onto the mean-zero
    subspace and cleared of denominators, so `sum(step) == 0` exactly.  The two
    alternatives are then `scale * q0 + step` and `scale * q0 - step`; both have
    total weight `scale * sum(q0)`, so both sit at l1 distance
    `sum|step| / (scale * sum(q0))` from the null.  `scale` is the smallest
    integer that keeps every weight positive and brings that distance at or below
    the requested one.
    """
    total = sum(q0)
    signs = [1 if i % 2 == 0 else -1 for i in range(len(q0))]
    imbalance = sum(s * q for s, q in zip(signs, q0))
    step = [q * (s * total - imbalance) for s, q in zip(signs, q0)]
    amplitude = sum(abs(s) for s in step)
    scale = max(-(-amplitude * distance.denominator
                  // (distance.numerator * total)),
                max(abs(s) // q + 1 for s, q in zip(step, q0)))
    plus = tuple(scale * q + s for q, s in zip(q0, step))
    minus = tuple(scale * q - s for q, s in zip(q0, step))
    common = gcd(*plus, *minus)
    return (tuple(v // common for v in plus),
            tuple(v // common for v in minus),
            Fraction(amplitude, scale * total))


def tail(distribution, total, alpha):
    """Exact critical value and the exact size attained there."""
    counts = [count for _, count in distribution]
    survival = total
    for index, count in enumerate(counts):
        survival -= count
        if Fraction(survival, total) <= alpha:
            return distribution[index][0], Fraction(survival, total)
    return distribution[-1][0], Fraction(0)


def exceedance(distribution, total, threshold):
    return Fraction(sum(count for value, count in distribution
                        if value > threshold), total)


def two_sided(distribution, total, alpha):
    """Equal-tailed exact rejection region, each tail held to alpha/2.

    A referee will object that a one-sided test on a statistic whose mean FALLS
    under the alternative is a straw man.  It is not enough to answer that in
    prose, so the two-sided d_u test is computed here exactly as well.  The
    region is {S <= low} union {S > high}; `low = -1` means the lower tail is
    empty, which happens whenever the null puts more than alpha/2 on the minimum.
    """
    half = alpha / 2
    low, mass = -1, Fraction(0)
    cumulative = Fraction(0)
    for value, count in distribution:
        cumulative += Fraction(count, total)
        if cumulative > half:
            break
        low, mass = value, cumulative
    high, upper = distribution[-1][0], Fraction(0)
    survival = Fraction(0)
    for value, count in reversed(distribution):
        if survival > half:
            break
        high, upper = value, survival
        survival += Fraction(count, total)
    return low, high, mass + upper


def likelihood_ordered(distribution, total, alpha):
    """Two-sided region by probability ordering: reject the least likely values.

    A referee may object that the equal-tailed region is a straw man and that a
    density-ordered region (reject values in increasing order of null
    probability until the mass reaches alpha) is the fair two-sided test.  Both
    are computed so the comparison is exact either way.  Returns the rejection
    set and its exact size.
    """
    ordered = sorted((count, value) for value, count in distribution)
    mass, region = Fraction(0), set()
    for count, value in ordered:
        step = Fraction(count, total)
        if mass + step > alpha:
            break
        mass += step
        region.add(value)
    return region, mass


def two_sided_rejection(distribution, total, low, high):
    return Fraction(sum(count for value, count in distribution
                        if value <= low or value > high), total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=[10, 20, 30, 40])
    parser.add_argument("--alpha", type=str, default="1/20")
    parser.add_argument("--distance", type=str, default="1/5",
                        help="target l1 distance, held equal across phases")
    args = parser.parse_args()
    alpha = Fraction(args.alpha)
    target = Fraction(args.distance)

    print(f"exact throughout, alpha = {alpha} = {float(alpha)}")
    print("alternative = null +/- the same mean-zero multiplicative period-2 "
          "comb,\nso the two phases are at EXACTLY equal l1 distance\n")

    for name, q0 in NULLS:
      plus, minus, distance = comb_pair(q0, target)
      for phase, q1 in (("+", plus), ("-", minus)):
        l0, l1 = sum(q0), sum(q1)
        print(f"{name}   phase {phase}  ->  q1={q1}   "
              f"l1 distance {float(distance):.6f}")
        print(f"{'N':>5} | {'CT crit':>11} {'size':>8} {'power':>8} "
              f"| {'DTV crit':>9} {'size':>8} {'power':>8} "
              f"| {'DTV2 region':>13} {'size':>8} {'power':>8} "
              f"| {'DTV2L size':>10} {'power':>8}")
        for N in args.N:
            centered_null = calculate_exact_dr_distribution(N, q0)
            centered_alt = calculate_exact_dr_distribution(N, q1, q0)
            crit_r, size_r = tail(centered_null, l0 ** N, alpha)
            power_r = exceedance(centered_alt, l1 ** N, crit_r)

            du_null = calculate_exact_dtv_distribution_nonuniform(N, list(q0))
            du_alt = calculate_exact_dtv_distribution_nonuniform(N, list(q1))
            crit_d, size_d = tail(du_null, l0 ** N, alpha)
            power_d = exceedance(du_alt, l1 ** N, crit_d)

            low, high, size_2 = two_sided(du_null, l0 ** N, alpha)
            power_2 = two_sided_rejection(du_alt, l1 ** N, low, high)
            region_l, size_l = likelihood_ordered(du_null, l0 ** N, alpha)
            power_l = Fraction(sum(c for v, c in du_alt if v in region_l), l1 ** N)

            flag = "  <-- power < size" if power_d < size_d else ""
            region = f"<={low},>{high}"
            print(f"{N:>5} | {crit_r:>11} {float(size_r):>8.6f} "
                  f"{float(power_r):>8.6f} | {crit_d:>9} {float(size_d):>8.6f} "
                  f"{float(power_d):>8.6f} | {region:>13} {float(size_2):>8.6f} "
                  f"{float(power_2):>8.6f} | {float(size_l):>10.6f} "
                  f"{float(power_l):>8.6f}{flag}")
        print()


if __name__ == "__main__":
    main()
