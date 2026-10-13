"""Verification of dr_exact.py and of the claims made about it.

1. total mass          sum of counts must be L**N,
2. uniform reduction   all q_i = 1 gives S = n * DTV, so the distribution must
                       equal ct.py's uniform DTV distribution stretched by n,
3. brute force         direct enumeration of every histogram with its
                       multinomial weight,
4. Monte Carlo         the exact CDF must agree with the empirical CDF of
                       multinomial samples to within Monte Carlo error,
5. S = L * d_r         the integer statistic really is L times the centered DTV
                       evaluated in floating point,
6. residue classes     c is invariant under q -> lambda*q, which is the content
                       of "independent of L itself" in the cost argument,
7. Proposition 1       Var[D_i] = N(p_i + p_{i-1} - d_i^2), and the sign-locking
                       bound Pr[sgn D_i != sgn d_i] <= exp(-N d_i^2 / 2) holds
                       where it binds.
"""

import itertools
import sys
from math import comb, factorial

import numpy as np

from du_exact import prepare_integer_weights
from dr_exact import (calculate_exact_dr_distribution,
                       statistic_upper_bound)


def brute_force(N, q):
    n = len(q)
    total_weight = sum(q)
    totals = {}
    for cut in itertools.combinations(range(N + n - 1), n - 1):
        histogram = []
        previous = -1
        for c in cut:
            histogram.append(c - previous - 1)
            previous = c
        histogram.append(N + n - 1 - previous - 1)
        multiplicity = factorial(N)
        weight = 1
        for count, weight_i in zip(histogram, q):
            multiplicity //= factorial(count)
            weight *= weight_i ** count
        value = sum(abs(total_weight * (histogram[i] - histogram[i - 1])
                        - N * (q[i] - q[i - 1])) for i in range(1, n))
        totals[value] = totals.get(value, 0) + multiplicity * weight
    return totals


def uniform_dtv_distribution(N, n):
    """Exact uniform DTV counts, the recurrence of ../ct/CombTest/code/ct.py."""
    previous = [[[0] * (2 * N + 2) for _ in range(N + 1)] for _ in range(N + 1)]
    current = [[[0] * (2 * N + 2) for _ in range(N + 1)] for _ in range(N + 1)]
    for i in range(N + 1):
        current[i][i][0] = comb(N, i)
    for _ in range(1, n):
        previous, current = current, previous
        for a in range(N + 1):
            for b in range(N + 1):
                row = current[a][b]
                for c in range(len(row)):
                    row[c] = 0
        for placed in range(N + 1):
            remaining = N - placed
            for last in range(placed + 1):
                for value in range(2 * placed + 2):
                    count = previous[placed][last][value]
                    if count:
                        for new_last in range(remaining + 1):
                            current[placed + new_last][new_last][
                                value + abs(new_last - last)] += (
                                count * comb(remaining, new_last))
    totals = {}
    for last in range(N + 1):
        for value in range(2 * N + 1):
            count = current[N][last][value]
            if count:
                totals[value] = totals.get(value, 0) + count
    return totals


def check(name, condition):
    print(f"  {'OK  ' if condition else 'FAIL'}  {name}")
    return condition


def main():
    ok = True

    print("1. total mass == L**N")
    for probs, N in ((( 1, 2, 1, 2, 1), 8), ((16, 8, 4, 2, 1, 1), 6),
                     ((1, 2, 3, 4, 3, 2, 1), 5), ((1, 1, 1, 1), 9)):
        q = prepare_integer_weights(probs)
        distribution = calculate_exact_dr_distribution(N, q)
        total = sum(count for _, count in distribution)
        ok &= check(f"q={q}, N={N}: sum={total} vs L**N={sum(q) ** N}",
                    total == sum(q) ** N)

    print("2. uniform reduction: S == n * DTV")
    for n, N in ((4, 8), (5, 7), (6, 6)):
        q = [1] * n
        distribution = calculate_exact_dr_distribution(N, q)
        centered = {value: count for value, count in distribution if count}
        reference = {n * value: count
                     for value, count in uniform_dtv_distribution(N, n).items()}
        ok &= check(f"n={n}, N={N}", centered == reference)

    print("3. brute force enumeration")
    for probs, N in (((1, 2, 1, 2, 1), 6), ((16, 8, 4, 2, 1, 1), 5),
                     ((1, 3, 1), 8)):
        q = prepare_integer_weights(probs)
        distribution = calculate_exact_dr_distribution(N, q)
        fast = {value: count for value, count in distribution if count}
        ok &= check(f"q={q}, N={N}", fast == brute_force(N, q))

    print("4. Monte Carlo agreement (max |exact CDF - empirical CDF|)")
    rng = np.random.default_rng(32)
    for probs, N, draws in (((1, 2, 1, 2, 1), 20, 400000),
                            ((16, 8, 4, 2, 1, 1), 15, 400000)):
        q = prepare_integer_weights(probs)
        total_weight = sum(q)
        p = np.array(q, dtype=float) / total_weight
        distribution = calculate_exact_dr_distribution(N, q)
        counts = np.array([count for _, count in distribution], dtype=object)
        mass = np.array([float(c) / float(total_weight ** N) for c in counts])
        exact_cdf = np.cumsum(mass)
        x = rng.multinomial(N, p, size=draws)
        step = N * np.diff(np.array(q))
        sampled = np.abs(total_weight * np.diff(x, axis=1) - step).sum(1)
        empirical = np.bincount(sampled, minlength=len(mass))[:len(mass)] / draws
        gap = float(np.max(np.abs(exact_cdf - np.cumsum(empirical))))
        tolerance = 5.0 / np.sqrt(draws)
        ok &= check(f"q={q}, N={N}: gap={gap:.5f} tol={tolerance:.5f}",
                    gap < tolerance)

    print("5. S == L * d_r (integer path vs floating point)")
    for probs, N in (((16, 8, 4, 2, 1, 1), 200), ((1, 2, 3, 4, 3, 2, 1), 500)):
        q = prepare_integer_weights(probs)
        total_weight = sum(q)
        p = np.array(q, dtype=float) / total_weight
        x = rng.multinomial(N, p, size=2000)
        centered = x - N * p
        float_value = np.abs(np.diff(centered, axis=1)).sum(1)
        integer_value = np.abs(total_weight * np.diff(x, axis=1)
                               - N * np.diff(np.array(q))).sum(1)
        gap = float(np.max(np.abs(float_value - integer_value / total_weight)))
        ok &= check(f"q={q}, N={N}: max |d_r - S/L| = {gap:.2e}", gap < 1e-9)

    print("6. residue classes are invariant under q -> lambda*q")
    for probs in ((1, 2, 1, 2, 1), (16, 8, 4, 2, 1, 1), (3, 1, 4)):
        reference = None
        for factor in (1, 2, 7, 145):
            q = [factor * v for v in probs]
            total_weight = sum(q)
            support = [v for v, c in
                       calculate_exact_dr_distribution(9, q) if c]
            signature = (len(support), len({v % total_weight for v in support}))
            reference = signature if reference is None else reference
            if signature != reference:
                ok &= check(f"q={probs} x{factor}: {signature} vs {reference}", False)
        ok &= check(f"q={probs}: support size and residue count fixed for "
                    f"lambda up to 145 (L up to {145 * sum(probs)})", True)

    print("7. Proposition 1: variance and the sign-locking bound")
    for probs, N in (((1, 2, 3, 4, 3, 2, 1), 400), ((16, 8, 4, 2, 1, 1), 200)):
        q = prepare_integer_weights(probs)
        p = np.array(q, dtype=float) / sum(q)
        step = np.diff(p)
        x = rng.multinomial(N, p, size=200000)
        differences = np.diff(x, axis=1)
        predicted = N * (p[1:] + p[:-1] - step ** 2)
        relative = float(np.max(np.abs(differences.var(0, ddof=1) - predicted) / predicted))
        ok &= check(f"q={q}, N={N}: variance formula, max relative error {relative:.3f}",
                    relative < 0.05)
    for p, N in ((np.array([0.2333, 0.3333, 0.4334]), 1000),
                 (np.array([0.10, 0.20, 0.30, 0.40]), 2000)):
        p = p / p.sum()
        step = np.diff(p)
        bound = float(np.sum(np.exp(-N * step ** 2 / 2.0)))
        x = rng.multinomial(N, p, size=200000)
        flipped = 1.0 - float(np.mean(np.all(np.sign(np.diff(x, axis=1)) == np.sign(step), axis=1)))
        ok &= check(f"p={np.round(p, 4)}, N={N}: Pr[not locked] {flipped:.2e} "
                    f"<= bound {bound:.2e}", flipped <= bound)

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
