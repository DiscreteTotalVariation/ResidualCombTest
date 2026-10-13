"""How biased can the comb test (CT) be?  An exhaustive, seed-free answer.

Good and Smith (1984) show that exact multinomial goodness-of-fit tests can be
biased: power below size against some alternatives.  The Letter reports the
CT's shortfall against period-2 combs.  An earlier version of that
sentence quoted the worst shortfall over RANDOMLY DRAWN nulls, which is a maximum
over a random sample and therefore moves with the seed: it read 0.0086 under one
seed and 0.0164 under another, with the amplitude at which it occurred moving
from 0.10 to 0.50.  A worst case must not be quoted from a random sample.

This script removes the randomness.  It enumerates COMPLETE families of rational
nulls -- every weight vector in a stated box, reduced to lowest terms and with
the uniform case dropped -- and computes exact size and exact power against every
period-2 comb on a fixed amplitude grid, in exact rational arithmetic.  The
answer is a property of the family, reproducible to the last digit by anyone, and
it cannot move with a seed because there is no seed.

Reported for each family and overall: the largest shortfall (size minus power),
where it occurs, and the same restricted to the amplitudes the Letter operates
in.  The ratio power/size is given alongside, because under heavy discreteness a
small absolute shortfall can still mean the test almost never rejects.
"""

import argparse
import itertools
from fractions import Fraction as F
from math import gcd

from du_exact import calculate_exact_dtv_distribution_nonuniform as du_law
from dr_exact import calculate_exact_dr_distribution as dr_law

# (bins, largest weight, sample sizes).  The boxes shrink as the number of bins
# grows so that every family stays exhaustively enumerable in minutes.
FAMILIES = [(3, 9, (12, 16, 20, 25, 30, 40)),
            (4, 6, (12, 16, 20, 25)),
            (5, 4, (12, 16, 20)),
            (6, 3, (12, 16, 20))]

# Amplitudes of the multiplicative period-2 comb, as exact rationals, both signs.
AMPLITUDES = [F(1, 100), F(1, 50), F(1, 25), F(1, 20), F(1, 10),
              F(1, 5), F(1, 4), F(1, 3), F(1, 2)]

LETTER_MAX_AMPLITUDE = F(1, 10)      # the regime the Letter's experiments use


def tail(distribution, total, alpha):
    """Smallest cut with exact level at most alpha, and the size it attains."""
    cumulative = 0
    for value, count in reversed(distribution):
        if F(cumulative + count, total) > alpha:
            return value, F(cumulative, total)
        cumulative += count
    return -1, F(cumulative, total)


def exceedance(distribution, total, threshold):
    return F(sum(c for v, c in distribution if v > threshold), total)


def families(n, top):
    """Every weight vector in [1, top]^n, in lowest terms, excluding uniform."""
    seen = []
    for q in itertools.product(range(1, top + 1), repeat=n):
        if len(set(q)) == 1:
            continue
        g = 0
        for v in q:
            g = gcd(g, v)
        if g != 1:
            continue
        seen.append(q)
    return seen


def comb(q0, sign, eps):
    """Multiplicative period-2 comb: alternate bins scaled by 1 +/- eps."""
    out = []
    for i, v in enumerate(q0):
        factor = 1 + (eps if i % 2 == 0 else -eps) * sign
        out.append(v * factor)
    denominator = 1
    for value in out:
        denominator = denominator * value.denominator // gcd(denominator, value.denominator)
    scaled = [int(value * denominator) for value in out]
    g = 0
    for v in scaled:
        g = gcd(g, v)
    return tuple(v // g for v in scaled) if g else tuple(scaled)


def scan(n, top, sample_sizes, alpha, verbose):
    """Worst shortfall of each test over one complete family."""
    worst = {"CT": (F(0), None), "du": (F(0), None)}
    worst_small = {"CT": (F(0), None), "du": (F(0), None)}
    counts = {"CT": [0, 0], "du": [0, 0]}       # [cases below size, cases]
    by_eps = {name: {eps: [0, 0, F(0)] for eps in AMPLITUDES} for name in counts}  # [below, cases, worst shortfall]
    nulls = families(n, top)
    for N in sample_sizes:
        for q0 in nulls:
            L0 = sum(q0)
            laws = {"CT": (dr_law(N, list(q0)), lambda q1: dr_law(N, list(q1), list(q0))),
                    "du": (du_law(N, list(q0)), lambda q1: du_law(N, list(q1)))}
            cuts = {}
            for name, (law0, _) in laws.items():
                cuts[name] = tail(law0, L0 ** N, alpha)
            for eps in AMPLITUDES:
                for sign in (1, -1):
                    q1 = comb(q0, sign, eps)
                    if min(q1) <= 0:
                        continue
                    for name, (_, under) in laws.items():
                        cut, size = cuts[name]
                        power = exceedance(under(q1), sum(q1) ** N, cut)
                        counts[name][1] += 1
                        by_eps[name][eps][1] += 1
                        if power < size:
                            counts[name][0] += 1
                            by_eps[name][eps][0] += 1
                            short = size - power
                            by_eps[name][eps][2] = max(by_eps[name][eps][2], short)
                            where = (q0, N, eps, sign, float(size), float(power))
                            if short > worst[name][0]:
                                worst[name] = (short, where)
                            if eps <= LETTER_MAX_AMPLITUDE and short > worst_small[name][0]:
                                worst_small[name] = (short, where)
        if verbose:
            print(f"    ... n={n}, N={N} done", flush=True)
    return worst, worst_small, counts, len(nulls), by_eps


def describe(label, entry):
    short, where = entry
    if where is None:
        return f"    {label:<28} none"
    q0, N, eps, sign, size, power = where
    ratio = power / size if size else float("nan")
    return (f"    {label:<28} {float(short):.4f}   at q={q0}, N={N}, "
            f"eps={float(eps):.2f}, sign={sign:+d}; size {size:.4f}, "
            f"power {power:.4f} (power/size {ratio:.2f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    alpha = F(args.alpha).limit_denominator(10 ** 6)

    print("Exhaustive, seed-free bias survey: exact size and exact power against every")
    print("period-2 comb on a fixed amplitude grid, over COMPLETE families of rational")
    print("nulls.  No randomness anywhere, so the worst case is a property of the family")
    print(f"and is reproducible to the last digit.  alpha = {args.alpha}.")
    print(f"Amplitudes: {', '.join(str(e) for e in AMPLITUDES)}, both signs.")
    print("power/size below 1 means the test is biased against that alternative.\n")

    grand = {"CT": (F(0), None), "du": (F(0), None)}
    grand_small = {"CT": (F(0), None), "du": (F(0), None)}
    grand_counts = {"CT": [0, 0], "du": [0, 0]}
    grand_eps = {name: {eps: [0, 0, F(0)] for eps in AMPLITUDES} for name in grand_counts}
    for n, top, sizes in FAMILIES:
        print(f"family: {n} bins, weights 1..{top}, N in {sizes}", flush=True)
        worst, worst_small, counts, n_nulls, by_eps = scan(n, top, sizes, alpha, not args.quiet)
        print(f"    {n_nulls} nulls in lowest terms, uniform excluded")
        for name in ("CT", "du"):
            below, total = counts[name]
            print(f"    {name}: below size in {below}/{total} "
                  f"({100.0 * below / max(total, 1):.1f}%)")
            print(describe(f"{name} worst shortfall", worst[name]))
            print(describe(f"{name} worst, eps <= {float(LETTER_MAX_AMPLITUDE):.2f}",
                           worst_small[name]))
            if worst[name][0] > grand[name][0]:
                grand[name] = worst[name]
            if worst_small[name][0] > grand_small[name][0]:
                grand_small[name] = worst_small[name]
            grand_counts[name][0] += below
            grand_counts[name][1] += total
            for eps, (b, t, w) in by_eps[name].items():
                grand_eps[name][eps][0] += b
                grand_eps[name][eps][1] += t
                grand_eps[name][eps][2] = max(grand_eps[name][eps][2], w)
        print(flush=True)

    print("OVERALL, all families pooled")
    for name in ("CT", "du"):
        below, total = grand_counts[name]
        print(f"  {name}: below size in {below}/{total} "
              f"({100.0 * below / max(total, 1):.1f}%) of (null, N, comb) cases")
        print(describe(f"{name} worst shortfall", grand[name]))
        print(describe(f"{name} worst, eps <= {float(LETTER_MAX_AMPLITUDE):.2f}",
                       grand_small[name]))
    print("\n  by amplitude (share of cases below size / worst shortfall), all families pooled:")
    print(f"  {'eps':>6} {'CT below':>10} {'CT worst':>10} {'du below':>10} {'du worst':>10}")
    for eps in AMPLITUDES:
        r, c = grand_eps["CT"][eps], grand_eps["du"][eps]
        print(f"  {float(eps):>6.2f} {100.0 * r[0] / max(r[1], 1):>9.1f}% {float(r[2]):>10.4f} "
              f"{100.0 * c[0] / max(c[1], 1):>9.1f}% {float(c[2]):>10.4f}")
    print("  (at the smallest amplitude power ~ size and the sign of the difference is a")
    print("   lattice artifact, so about half the cases fall below size for ANY test)")
    print("\nThe Letter should quote these numbers, not a worst case over randomly")
    print("drawn nulls: those move with the seed (0.0086 -> 0.0164 on a seed change).")


if __name__ == "__main__":
    main()
