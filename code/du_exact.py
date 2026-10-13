import argparse
import sys
from fractions import Fraction as F
from functools import reduce
from math import comb, gcd
from os import makedirs
from os.path import dirname

# Efficient (dynamic-programming) exact DTV distribution for a NON-UNIFORM null
# distribution. This combines:
#   * the bin-by-bin DTV recurrence of ../ct/CombTest/code/ct.py, which tracks
#     counts[sum][last][dtv] = number of ways the first processed bins hold
#     `sum` samples, end with `last` samples in the current bin, and give a
#     running discrete total variation of `dtv`; and
#   * the non-uniform weighting of ../zds_vu/code/exact_chi_squared.py, where
#     each bin transition is multiplied by q[i]**(count placed in bin i).
#
# As explained in ../zds_vu/paper, the probabilities p_i = a_i/b_i are converted
# to integer weights q_i = p_i*L (L = lcm of denominators). The DP then produces
# the same counts as the brute-force enumeration (du_exact_bf.py / du_exact_bf2.py),
# summing to L**N, but in polynomial time. When all q_i = 1 (uniform case) it
# reduces exactly to ct.py, with counts summing to n**N.
#
# DTV = sum_{i=1}^{n-1} |h_i - h_{i-1}|.


def prepare_integer_weights(ps, max_denominator=10**9):
    """Convert a probability (or proportional weight) list to integer weights
    via the LCM of denominators, exactly as in zds_vu/code/exact_chi_squared.py."""
    fractions = [F(f).limit_denominator(max_denominator=max_denominator) for f in ps]

    def lcm(a, b):
        return a * b // gcd(a, b)

    denominators = [f.denominator for f in fractions]
    common_multiple = reduce(lcm, denominators)
    integer_weights = [int(f * common_multiple) for f in fractions]
    return integer_weights


def calculate_exact_dtv_distribution_nonuniform(N, q):
    n = len(q)
    if n == 0:
        return []

    # Two count buffers (swapped to save memory), each counts[sum][last][dtv].
    previous_counts = []
    current_counts = []
    for _ in range(N + 1):
        previous_counts.append([[0] * (2 * N + 1 + 1) for _ in range(N + 1)])
        current_counts.append([[0] * (2 * N + 1 + 1) for _ in range(N + 1)])

    # First bin (index 0): place i samples here, weighted by q[0]**i.
    for i in range(N + 1):
        current_counts[i][i][0] = comb(N, i) * (q[0] ** i)

    # Process the remaining bins one at a time.
    for bin_idx in range(1, n):
        previous_counts, current_counts = current_counts, previous_counts
        # Clear the current buffer.
        for a in range(N + 1):
            for b in range(N + 1):
                row = current_counts[a][b]
                for c in range(len(row)):
                    row[c] = 0

        for previous_sum in range(0, N + 1):
            remaining = N - previous_sum
            for previous_last in range(0, previous_sum + 1):
                for previous_dtv in range(0, 2 * previous_sum + 1 + 1):
                    previous_count = previous_counts[previous_sum][previous_last][previous_dtv]
                    if previous_count > 0:
                        for new_last in range(0, remaining + 1):
                            new_sum = previous_sum + new_last
                            new_dtv = previous_dtv + abs(new_last - previous_last)
                            current_counts[new_sum][new_last][new_dtv] += (
                                previous_count
                                * comb(remaining, new_last)
                                * (q[bin_idx] ** new_last)
                            )

    # Collect the final distribution from states with all N samples placed.
    dtv_counts = dict()
    for last in range(0, N + 1):
        for dtv in range(0, 2 * N + 1):
            count = current_counts[N][last][dtv]
            if count:
                dtv_counts[dtv] = dtv_counts.get(dtv, 0) + count

    # Pad with trailing zeros up to dtv = 2*N, matching ct.py's output format.
    distribution = []
    for dtv in range(0, 2 * N + 1):
        distribution.append((dtv, dtv_counts.get(dtv, 0)))
    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Efficient (DP) exact DTV distribution for a non-uniform null."
    )
    parser.add_argument("-N", type=int, required=True, help="Sample size")
    parser.add_argument("--probs", type=float, nargs="+", required=True,
                        help="Per-bin probabilities (or proportional weights); "
                             "the number of bins n is inferred from this list")
    parser.add_argument("--max_denominator", type=int, default=10**9,
                        help="Maximum denominator for rational approximation")
    parser.add_argument("-o", dest="output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    q = prepare_integer_weights(args.probs, max_denominator=args.max_denominator)
    distribution = calculate_exact_dtv_distribution_nonuniform(args.N, q)

    lines = ["%d %d" % (dtv, count) for dtv, count in distribution]
    text = "\n".join(lines) + "\n"

    if args.output is None:
        sys.stdout.write(text)
    else:
        out_dir = dirname(args.output)
        if out_dir:
            makedirs(out_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf8") as fo:
            fo.write(text)


if __name__ == "__main__":
    main()
