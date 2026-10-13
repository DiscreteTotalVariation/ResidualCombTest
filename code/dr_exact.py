import argparse
import sys
from math import comb
from os import makedirs
from os.path import dirname

from du_exact import prepare_integer_weights

# Exact null distribution of the CENTERED discrete total variation d_r, the
# statistic of the comb test (CT),
#
#   d_r(x) = sum_{i=2}^{n} |e_i - e_{i-1}|,   e_i = x_i - N p_i (centered counts),
#
# for a non-uniform null p_i = q_i / L with integer weights q_i and L = sum q_i.
#
# d_r is not integer-valued, but L * d_r is:
#
#   S(x) = L * d_r(x) = sum_{i=2}^{n} | L (x_i - x_{i-1}) - N (q_i - q_{i-1}) |,
#
# so the whole computation stays in exact integer arithmetic on the lattice 1/L,
# with no discretisation error.  This is the point of centering: d_r
# keeps the exactness of the uniform-null Comb Test while removing the baseline
# slope that makes the uncentered DTV d_u degenerate.
#
# The recurrence is that of du_exact.py with the accumulated statistic replaced by
# S; the extra state is still only the previous bin count, so the statistic is
# handled without being decomposable.  The RANGE of S is O(L*N) rather than the
# O(N) of the uncentered d_u, but the SUPPORT is not: every increment
# |L*(x_i - x_{i-1}) - N*(q_i - q_{i-1})| is congruent to +-N*(q_i - q_{i-1})
# mod L, so S mod L takes at most c <= min(L, 2^(n-1)) values.  c is set by q
# and by N (through N*(q_i - q_{i-1}) mod L, i.e. by N mod L): for the
# run-length null q=(16,8,4,2,1,1) it is 8, 16, 4, 16, 2, 4 at N = 10, 15, 20,
# 25, 40, 100.  It does not grow with a common scaling q -> lambda*q, which
# multiplies L.  The reachable values therefore lie on c translates of the
# lattice L*Z and their count is O(c*N); measured c = 1 to 16 for the nulls of
# the paper at N <= 100 (results/experiment_dp_complexity.txt), and the
# support is literally the same (182 values at N=20) for q=(16,8,4,2,1,1) at
# L=32 and at L=4640, where the range of S differs by a factor 145.  The
# dict-of-dicts DP visits only reachable values, so the cost is that of du_exact.py
# times a small constant set by c (measured 1.1-1.6 for L<=41 and 3.3 at
# L=4640, where big-integer digit length grows like log L), not times L.  Only
# the zero-padded output list is O(L*N).  See experiment_dp_complexity.py.
# (An earlier version of this comment said "a fixed residue class"; that was
# wrong -- the absolute value can flip the sign of each increment.)
#
# Self-check: when all q_i = 1 we have L = n and q_i - q_{i-1} = 0, so
# S = n * DTV and the output is the uniform DTV distribution of ct.py with its
# support stretched by a factor n.


def statistic_upper_bound(N, q):
    """Largest value L*d_r can take: L*max(DTV) + N*sum|q_i - q_{i-1}|."""
    total_weight = sum(q)
    slope = sum(abs(q[i] - q[i - 1]) for i in range(1, len(q)))
    return 2 * N * total_weight + N * slope


def calculate_exact_dr_distribution(N, q, q_centre=None, raw=False):
    """Counts of L*d_r over all L**N weighted histograms of N samples.

    `q` gives the sampling weights.  `q_centre` gives the weights the statistic
    is centred on and defaults to `q`, which is the null.  Passing an alternative
    as `q` while holding `q_centre` at the null yields the exact distribution of
    the statistic under that alternative, and hence exact power with no Monte
    Carlo error anywhere in the comparison.

    `raw=True` runs the SAME program on the uncentered statistic, L * DTV: the
    increment becomes L*|m - k| and the weights stay q.  Its only use is a
    like-for-like timing against the exact DP of d_r (experiment_timing.py --fair);
    du_exact.py is the reference implementation of the uncentered law.  (The
    keyword keeps its short name `raw` because experiment_timing.py passes it.)
    """
    n = len(q)
    if n == 0:
        return []
    if q_centre is None:
        q_centre = q
    total_weight = sum(q_centre)

    # previous[(placed, last)] maps an accumulated L*d_r value to its count.
    previous = {}
    for i in range(N + 1):
        previous[(i, i)] = {0: comb(N, i) * (q[0] ** i)}

    for bin_index in range(1, n):
        weight_step = 0 if raw else N * (q_centre[bin_index] - q_centre[bin_index - 1])
        current = {}
        for (placed, last), accumulated in previous.items():
            remaining = N - placed
            for new_last in range(remaining + 1):
                term = abs(total_weight * (new_last - last) - weight_step)
                multiplier = comb(remaining, new_last) * (q[bin_index] ** new_last)
                target = current.setdefault((placed + new_last, new_last), {})
                for value, count in accumulated.items():
                    shifted = value + term
                    target[shifted] = target.get(shifted, 0) + count * multiplier
        previous = current

    totals = {}
    for (placed, _), accumulated in previous.items():
        if placed != N:
            continue
        for value, count in accumulated.items():
            totals[value] = totals.get(value, 0) + count

    upper = max(statistic_upper_bound(N, q_centre), max(totals) if totals else 0)
    return [(value, totals.get(value, 0)) for value in range(upper + 1)]


def main():
    parser = argparse.ArgumentParser(
        description="Exact distribution of S = L*d_r (the centered DTV, the "
                    "comb test's statistic) for a non-uniform null.")
    parser.add_argument("-N", type=int, required=True, help="Sample size")
    parser.add_argument("--probs", type=float, nargs="+", required=True,
                        help="Per-bin probabilities or proportional weights")
    parser.add_argument("--centre", type=float, nargs="+", default=None,
                        help="Weights the statistic is centred on; defaults to "
                             "--probs. Give the null here and an alternative in "
                             "--probs to obtain the exact distribution under "
                             "that alternative.")
    parser.add_argument("--max_denominator", type=int, default=10 ** 9,
                        help="Maximum denominator for rational approximation")
    parser.add_argument("-o", dest="output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    q = prepare_integer_weights(args.probs, max_denominator=args.max_denominator)
    centre = None
    if args.centre is not None:
        centre = prepare_integer_weights(args.centre,
                                         max_denominator=args.max_denominator)
    distribution = calculate_exact_dr_distribution(args.N, q, centre)

    text = "\n".join("%d %d" % pair for pair in distribution) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        out_dir = dirname(args.output)
        if out_dir:
            makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
