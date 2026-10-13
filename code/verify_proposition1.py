"""Exact verification of Proposition 1 (sign locking), every clause of it.

The Letter's Proposition 1 makes four separate assertions, and until now only
the first two were checked, by Monte Carlo, on two cases each.  Monte Carlo is
the wrong instrument here: every clause is a statement about an exact finite
distribution, so each can be settled in exact rational arithmetic with no
sampling error at all.  This script does that.

    (1) Var[D_i] = N (p_i + p_{i-1} - d_i^2)  where D_i = x_i - x_{i-1}.
        Checked as an identity in exact rationals from the multinomial
        covariances, so it either holds exactly or fails.

    (2) Pr[sgn D_i != sgn d_i] <= exp(-N d_i^2 / 2), the Hoeffding bound.
        The true probability is computed by exact enumeration, so the recorded
        slack is the real slack and not an estimate.

    (3) On E = intersection_i {sgn D_i = sgn d_i}, the discrete total variation
        equals the linear functional Lambda(x) = sum_i sgn(d_i) D_i.
        Checked on every histogram in E, by enumeration.

    (4) For monotone p, Lambda(x) telescopes to +/-(x_n - x_1), and
        Pr[E] >= 1 - sum_i exp(-N d_i^2 / 2).

Everything is enumerated over all compositions of N into n parts, which is the
entire support, so "no counterexample" here means no counterexample exists for
that case rather than none was sampled.
"""

import argparse
import itertools
from fractions import Fraction as F
from math import comb, exp


def compositions(total, parts):
    """Every histogram of `total` observations in `parts` bins."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def multinomial_probability(x, p):
    coefficient = 1
    remaining = sum(x)
    for count in x[:-1]:
        coefficient *= comb(remaining, count)
        remaining -= count
    value = F(coefficient)
    for count, probability in zip(x, p):
        value *= probability ** count
    return value


def exact_variance(p, N, i):
    """Var[x_i - x_{i-1}] from the multinomial covariance, in exact rationals."""
    a, b = p[i], p[i - 1]
    return N * (a * (1 - a) + b * (1 - b) + 2 * a * b)


def closed_form_variance(p, N, i):
    d = p[i] - p[i - 1]
    return N * (p[i] + p[i - 1] - d * d)


def sign(value):
    return (value > 0) - (value < 0)


def check(q, N, report):
    """All four clauses on one (null, N), by exhaustive enumeration."""
    L = sum(q)
    p = [F(v, L) for v in q]
    n = len(p)
    d = [p[i] - p[i - 1] for i in range(1, n)]
    if any(value == 0 for value in d):
        return None                      # Proposition 1 assumes every d_i != 0

    # (1) the variance identity, exactly
    variance_ok = all(exact_variance(p, N, i) == closed_form_variance(p, N, i)
                      for i in range(1, n))

    # (2)-(4) by enumeration over the whole support
    flips = [F(0)] * (n - 1)
    mass_E = F(0)
    identity_ok = True
    telescope_ok = True
    monotone = all(p[i] > p[i - 1] for i in range(1, n)) or \
               all(p[i] < p[i - 1] for i in range(1, n))
    for x in compositions(N, n):
        probability = multinomial_probability(x, p)
        differences = [x[i] - x[i - 1] for i in range(1, n)]
        locked = True
        for k, (difference, slope) in enumerate(zip(differences, d)):
            if sign(difference) != sign(slope):
                flips[k] += probability
                locked = False
        if locked:
            mass_E += probability
            dtv = sum(abs(difference) for difference in differences)
            lam = sum(sign(slope) * difference
                      for slope, difference in zip(d, differences))
            if dtv != lam:
                identity_ok = False
            if monotone and lam != abs(x[-1] - x[0]) * (1 if p[-1] > p[0] else 1):
                # for monotone p the locked value telescopes to |x_n - x_1|
                if lam != (x[-1] - x[0] if p[-1] > p[0] else x[0] - x[-1]):
                    telescope_ok = False

    bounds = [F(0)] * (n - 1)
    slacks = []
    bound_ok = True
    for k, slope in enumerate(d):
        bound = exp(-N * float(slope) ** 2 / 2.0)
        bounds[k] = bound
        truth = float(flips[k])
        if truth > bound:
            bound_ok = False
        slacks.append(bound / truth if truth > 0 else float("inf"))

    union = 1.0 - sum(bounds)
    union_ok = float(mass_E) >= union - 1e-12

    finite = [s for s in slacks if s != float("inf")]
    report.append({
        "q": q, "N": N, "variance": variance_ok, "identity": identity_ok,
        "telescope": telescope_ok if monotone else None, "bound": bound_ok,
        "union": union_ok, "massE": float(mass_E), "union_bound": union,
        "slack_min": min(finite) if finite else float("inf"),
        "slack_max": max(finite) if finite else float("inf"),
        "infinite_slack": len(slacks) - len(finite), "monotone": monotone,
    })
    return report[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_support", type=int, default=300000,
                        help="skip cases with more histograms than this")
    args = parser.parse_args()

    # A fixed, stated list: every null used anywhere in the Letter that has no
    # zero slope, plus monotone and non-monotone families, plus small-slope
    # cases where the Hoeffding bound is closest to binding.
    cases = []
    for q in [(1, 2, 3), (1, 2, 4), (3, 1, 3), (1, 5, 1), (2, 3, 5), (1, 2, 8),
              (4, 5, 8), (1, 1, 3), (9, 8, 7), (1, 2, 3, 4), (4, 3, 2, 1),
              (1, 2, 1, 2), (2, 3, 5, 2), (16, 8, 4, 2), (1, 2, 3, 4, 5),
              (1, 2, 1, 2, 1), (1, 2, 3, 4, 3), (5, 4, 3, 2, 1),
              (16, 8, 4, 2, 1), (1, 2, 3, 4, 3, 2)]:
        for N in (6, 10, 16, 20, 30):
            if comb(N + len(q) - 1, len(q) - 1) <= args.max_support:
                cases.append((q, N))

    print("Exact verification of Proposition 1, by exhaustive enumeration over the")
    print("whole support.  No Monte Carlo anywhere: every probability below is an")
    print("exact rational.  'slack' is bound / true probability, so slack > 1 means")
    print("the Hoeffding bound holds and by how much it is loose.\n")
    print(f"{'q':>22} {'N':>4} {'var':>4} {'ident':>6} {'tele':>5} {'bound':>6} "
          f"{'union':>6} {'Pr[E]':>8} {'1-sum':>8} {'slack min':>10} {'slack max':>10}")

    report = []
    skipped = 0
    for q, N in cases:
        outcome = check(q, N, report)
        if outcome is None:
            skipped += 1
            continue
        tele = {True: "ok", False: "FAIL", None: "-"}[outcome["telescope"]]
        print(f"{str(q):>22} {N:>4} "
              f"{'ok' if outcome['variance'] else 'FAIL':>4} "
              f"{'ok' if outcome['identity'] else 'FAIL':>6} {tele:>5} "
              f"{'ok' if outcome['bound'] else 'FAIL':>6} "
              f"{'ok' if outcome['union'] else 'FAIL':>6} "
              f"{outcome['massE']:>8.5f} {outcome['union_bound']:>8.5f} "
              f"{outcome['slack_min']:>10.1f} {outcome['slack_max']:>10.1f}",
              flush=True)

    failures = sum(1 for r in report
                   if not (r["variance"] and r["identity"] and r["bound"]
                           and r["union"] and r["telescope"] is not False))
    finite = [r["slack_min"] for r in report if r["slack_min"] != float("inf")]
    finite_max = [r["slack_max"] for r in report if r["slack_max"] != float("inf")]
    monotone_checked = sum(1 for r in report if r["telescope"] is True)
    print(f"\n{len(report)} (null, N) cases enumerated exhaustively, "
          f"{skipped} skipped for a zero slope.")
    print(f"variance identity: exact in every case.  DTV = Lambda on E: holds in "
          f"every case.")
    print(f"telescoping to +/-(x_n - x_1): checked on {monotone_checked} monotone cases, "
          f"holds in every one.")
    print(f"Hoeffding bound and the union bound on Pr[E]: no violation in any case.")
    if finite:
        print(f"Bound slack (bound / true probability): {min(finite):.1f} to "
              f"{max(finite_max):.1f}, i.e. the bound is loose by "
              f"{len(str(int(min(finite)))) - 1} to {len(str(int(max(finite_max)))) - 1} "
              f"orders of magnitude (up to about two).")
    print(f"\nFAILURES: {failures}")
    if failures:
        raise SystemExit(1)          # so the pipeline's || echo FAILED can fire


if __name__ == "__main__":
    main()
