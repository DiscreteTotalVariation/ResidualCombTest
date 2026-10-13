"""Can a test's power fall BELOW its own size, and how often?

A one-sided test on a statistic whose null expectation is not its minimum can be
biased in the sense of Good and Smith (1984): for some alternatives it rejects
less often than under the null.  Table I of the letter shows two such cells for
the uncentered d_u.  This script measures how common the region is, for BOTH tests, by
exact computation over a whole lattice of alternatives: for a null q0 with
L = sum q0, every integer weight vector q1 > 0 with the same L is an alternative
p1 = q1/L, and the DP gives the exact size and the exact power, with no Monte
Carlo anywhere.  Lattices larger than --cap are sampled at random with a fixed
seed.

Findings (see results/experiment_bias.txt):
  * the uncentered d_u is biased for a large fraction of alternatives on every
    non-uniform null tried -- 11 of the 14 on the comb null q=(1,2,1,2,1), and
    all 14 on q=(1,5,1) -- with shortfalls down to power 0.0000 against a size
    near 0.05;
  * against PERIOD-2 COMBS, the class the test is for, the uncentered d_u's power
    falls as low as 0.0006 against a size of 0.0465 (comb null, eps = 0.25),
    while the CT's shortfall never exceeds 0.004 and occurs only at
    eps <= 0.05, where neither test has appreciable power yet;
  * over the full lattice the comb test (CT) is far less prone to it: none of the alternatives on
    six of the seven non-uniform nulls, and on the tent null only far-away
    alternatives, the region shrinking quickly with N (357 of 5004 at N=10,
    12 of 5004 at N=16, none of them within l1 = 0.25 of the null).
The mechanism is a DISPERSION effect, not a mean effect: the exact curves at the
end of the output show that at the worst amplitude the mean of the statistic is at
or above its null value while its standard deviation is 1 to 3 per cent below, so
the upper tail moves down even as the centre moves up.  An earlier guess that the
mean drifts down is wrong for two of the three nulls shown.

So the repair is not a proof of unbiasedness, and the letter does not claim one;
within the period-2, small-l1 regime the letter operates in, the CT's
power stayed at or above its size in every case computed here.
"""

import argparse
import itertools
import random
from fractions import Fraction as F

from du_exact import calculate_exact_dtv_distribution_nonuniform as du_law
from dr_exact import calculate_exact_dr_distribution as dr_law

NULLS = [((1, 2, 1, 2, 1), 40), ((1, 2, 1, 2, 1), 20), ((16, 8, 4, 2, 1, 1), 12),
         ((1, 1, 1, 1, 1), 30), ((1, 2, 3, 4, 3, 2, 1), 10), ((1, 2, 3, 4, 3, 2, 1), 16),
         ((3, 1, 3), 40), ((1, 5, 1), 30), ((2, 3, 5, 2), 24), ((1, 1, 4, 1, 1), 25)]


def tail(distribution, total, alpha):
    """Smallest s with Pr[S > s] <= alpha, and the size attained there."""
    survival = total
    for value, count in distribution:
        survival -= count
        if F(survival, total) <= alpha:
            return value, F(survival, total)
    return distribution[-1][0], F(0)


def exceedance(distribution, total, threshold):
    return F(sum(c for v, c in distribution if v > threshold), total)


def compositions(total, parts):
    for cut in itertools.combinations(range(1, total), parts - 1):
        edges = (0,) + cut + (total,)
        yield tuple(edges[i + 1] - edges[i] for i in range(parts))


def survey(q0, N, alpha, cap, seed):
    L, n = sum(q0), len(q0)
    crit_r, size_r = tail(dr_law(N, list(q0)), L ** N, alpha)
    crit_w, size_w = tail(du_law(N, list(q0)), L ** N, alpha)
    every = [q for q in compositions(L, n) if q != tuple(q0)]
    exhaustive = len(every) <= cap
    alts = every if exhaustive else random.Random(seed).sample(every, cap)
    fails = {"CT": [], "du": []}
    for q1 in alts:
        l1 = float(sum(abs(F(a, L) - F(b, L)) for a, b in zip(q1, q0)))
        power_r = exceedance(dr_law(N, list(q1), list(q0)), sum(q1) ** N, crit_r)
        power_w = exceedance(du_law(N, list(q1)), sum(q1) ** N, crit_w)
        if power_r < size_r:
            fails["CT"].append((l1, q1, float(power_r)))
        if power_w < size_w:
            fails["du"].append((l1, q1, float(power_w)))
    kind = "exhaustive" if exhaustive else f"random {cap} of {len(every)}"
    print(f"null q={q0}, L={L}, N={N}, alpha={float(alpha)}: {len(alts)} alternatives ({kind})")
    for name, size in (("CT", size_r), ("du", size_w)):
        bad = sorted(fails[name])
        line = (f"  {name:>3}  size {float(size):.4f}: power below size in "
                f"{len(bad)}/{len(alts)}")
        if bad:
            near = sum(1 for f in bad if f[0] <= 0.25)
            line += (f"; smallest l1 {bad[0][0]:.3f} (power {bad[0][2]:.4f}); "
                     f"deepest power {min(f[2] for f in bad):.4f}; "
                     f"{near} of them within l1 = 0.25")
        print(line)


def comb_weights(q0, sign, numerator, denominator):
    """Integer weights of the multiplicative period-2 comb q_i -> q_i(1 +- eps)."""
    return tuple(v * (denominator + (sign if i % 2 == 0 else -sign) * numerator)
                 for i, v in enumerate(q0))


def comb_survey(q0, N, alpha):
    """The same question restricted to the alternative class the test is FOR.
    The lattice survey above ranges over every alternative, most of which no comb
    test is meant to catch; here only period-2 combs of both signs are used, over
    a range of eps, and the SHORTFALL is reported, not just the count."""
    L0 = sum(q0)
    crit_r, size_r = tail(dr_law(N, list(q0)), L0 ** N, alpha)
    crit_w, size_w = tail(du_law(N, list(q0)), L0 ** N, alpha)
    worst = {"CT": (0.0, None), "du": (0.0, None)}
    counts = {"CT": 0, "du": 0}
    total = 0
    for denominator, numerator in ((100, 2), (50, 2), (20, 1), (100, 10), (10, 2), (4, 1), (2, 1)):
        for sign in (+1, -1):
            q1 = comb_weights(q0, sign, numerator, denominator)
            if min(q1) <= 0:
                continue
            L1 = sum(q1)
            l1 = float(sum(abs(F(a, L1) - F(b, L0)) for a, b in zip(q1, q0)))
            eps = numerator / denominator
            total += 1
            for name, law, crit, size in (("CT", dr_law(N, list(q1), list(q0)), crit_r, size_r),
                                          ("du", du_law(N, list(q1)), crit_w, size_w)):
                power = exceedance(law, L1 ** N, crit)
                if power < size:
                    counts[name] += 1
                    shortfall = float(size - power)
                    if shortfall > worst[name][0]:
                        worst[name] = (shortfall, (eps, l1, float(power), float(size)))
    print(f"null q={q0}, N={N}: {total} period-2 combs (both signs, eps from 0.02 to 0.5)")
    for name in ("CT", "du"):
        line = f"  {name:>3}  power below size in {counts[name]}/{total}"
        if worst[name][1]:
            eps, l1, power, size = worst[name][1]
            line += (f"; largest shortfall {worst[name][0]:.4f} "
                     f"(power {power:.4f} vs size {size:.4f}) at eps={eps:.2f}, l1={l1:.3f}")
        print(line)


def moments(distribution, total):
    mean = sum(F(v * c, total) for v, c in distribution if c)
    second = sum(F(v * v * c, total) for v, c in distribution if c)
    return mean, second - mean * mean


def curve(q0, N, alpha):
    """WHY the CT dips below its size, settled exactly.

    Two mechanisms could produce power below size: the statistic's mean drifting
    DOWN as the comb grows, or its dispersion SHRINKING enough to pull the upper
    tail down even while the mean rises.  Monte Carlo cannot separate them
    reliably at these magnitudes, so here the exact power, the exact mean and the
    exact standard deviation are computed on a grid of rational amplitudes.  The
    answer is the second mechanism: at the worst amplitude the mean is at or above
    its null value while the standard deviation is 1 to 3 per cent below it.
    """
    L0 = sum(q0)
    r0 = dr_law(N, list(q0))
    w0 = du_law(N, list(q0))
    crit_r, size_r = tail(r0, L0 ** N, alpha)
    crit_w, size_w = tail(w0, L0 ** N, alpha)
    mean_0, var_0 = moments(r0, L0 ** N)
    print(f"null q={q0}, N={N}: CT rejects S>{crit_r}, exact size {float(size_r):.4f}; "
          f"du rejects >{crit_w}, exact size {float(size_w):.4f}")
    print(f"{'eps':>7} {'CT power':>10} {'power-size':>11} {'E[S]/E0':>9} {'sd/sd0':>8} "
          f"{'du power':>10} {'power-size':>11}")
    for den, num in ((200, 1), (100, 1), (100, 2), (50, 2), (100, 5), (20, 1),
                     (100, 10), (10, 2), (5, 1), (4, 1)):
        q1 = comb_weights(q0, +1, num, den)
        if min(q1) <= 0:
            continue
        L1 = sum(q1)
        r1 = dr_law(N, list(q1), list(q0))
        w1 = du_law(N, list(q1))
        power_r = exceedance(r1, L1 ** N, crit_r)
        power_w = exceedance(w1, L1 ** N, crit_w)
        mean_1, var_1 = moments(r1, L1 ** N)
        sd_ratio = float((var_1 / var_0) ** 0.5) if var_1 > 0 and var_0 > 0 else 0.0
        print(f"{num / den:>7.3f} {float(power_r):>10.5f} {float(power_r - size_r):>+11.5f} "
              f"{float(mean_1 / mean_0):>9.5f} {sd_ratio:>8.5f} "
              f"{float(power_w):>10.5f} {float(power_w - size_w):>+11.5f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--cap", type=int, default=6000,
                        help="lattices larger than this are sampled at random; 6000 keeps "
                             "every null here exhaustive except q=(16,8,4,2,1,1)")
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    print("Exact size and power over a lattice of alternatives; power < size means "
          "the test is biased\nagainst that alternative (Good and Smith 1984).  "
          "The letter's own regime is a period-2 comb\nat l1 about 0.1 to 0.2.\n")
    alpha = F(args.alpha).limit_denominator(10 ** 6)
    for q0, N in NULLS:
        survey(q0, N, alpha, args.cap, args.seed)
    print("\nRestricted to period-2 combs, the class the comb test is designed for:\n")
    for q0, N in NULLS:
        if sum(q0) == len(q0):          # the uniform control has no comb lattice here
            continue
        comb_survey(q0, N, alpha)
    print("\nMechanism, exactly: power, mean and standard deviation against amplitude\n")
    for q0, N in (((16, 8, 4, 2, 1, 1), 12), ((1, 1, 3), 30), ((4, 5, 8), 25)):
        curve(q0, N, alpha)


if __name__ == "__main__":
    main()
