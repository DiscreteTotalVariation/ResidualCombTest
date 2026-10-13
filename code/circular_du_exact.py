import argparse
import sys
from fractions import Fraction as F
from functools import reduce
from math import comb, gcd
from os import makedirs
from os.path import dirname

# Efficient (dynamic-programming) exact CIRCULAR DTV distribution for a
# NON-UNIFORM null distribution. This is to cct.py what du_exact.py is to ct.py:
# it combines
#   * the circular DTV dynamic program of ../ct/code/cct.py, which adds the
#     wrap-around edge |h_1 - h_n| by running the linear bin-by-bin DP once per
#     fixed first-bin value f and closing the cycle with |last - f| at the end;
#     and
#   * the non-uniform weighting of du_exact.py / ../zds_vu, where each bin
#     transition is multiplied by q[i]**(count placed in bin i).
#
# As explained in ../zds_vu/paper, the probabilities p_i = a_i/b_i are converted
# to integer weights q_i = p_i*L (L = lcm of denominators). The DP then produces
# the same counts as a brute-force enumeration, summing to L**N, but in
# polynomial time. When all q_i = 1 (uniform case) it reduces exactly to cct.py,
# with counts summing to n**N.
#
# Circular DTV of an n-bin histogram h is
#     CDTV = sum_{i=1}^{n-1} |h_{i+1} - h_i| + |h_1 - h_n|,
# i.e. the ordinary (linear) DTV plus the wrap-around edge between first and last
# bin. Its maximum value is 2*N.


def prepare_integer_weights(ps, max_denominator=10**9):
    """Convert a probability (or proportional weight) list to integer weights
    via the LCM of denominators, exactly as in du_exact.py / zds_vu."""
    fractions = [F(f).limit_denominator(max_denominator=max_denominator) for f in ps]

    def lcm(a, b):
        return a * b // gcd(a, b)

    denominators = [f.denominator for f in fractions]
    common_multiple = reduce(lcm, denominators)
    integer_weights = [int(f * common_multiple) for f in fractions]
    return integer_weights


def calculate_exact_circular_dtv_distribution_nonuniform(N, q):
    n = len(q)
    if n == 0:
        return []

    size_dtv = 2 * N + 1 + 1

    # state[sum][last][dtv]: weighted number of assignments of `sum` samples into
    # the bins placed so far (with the first bin fixed to f), ending with `last`
    # samples in the current bin and accumulated open (non-wrapped) DTV `dtv`.
    def make_table():
        return [[[0] * size_dtv for _ in range(N + 1)] for _ in range(N + 1)]

    def clear(table):
        for i in range(N + 1):
            for j in range(N + 1):
                row = table[i][j]
                for k in range(size_dtv):
                    row[k] = 0

    previous_counts = make_table()
    current_counts = make_table()

    # result[cdtv] -> weighted count, accumulated over all first-bin values f.
    result = dict()

    def readout(table, f):
        # close the cycle: every full configuration (sum == N) contributes its
        # open DTV plus the wrap-around edge |last - f|.
        for last in range(N + 1):
            row = table[N][last]
            wrap = abs(last - f)
            for dtv in range(size_dtv):
                count = row[dtv]
                if count:
                    cdtv = dtv + wrap
                    result[cdtv] = result.get(cdtv, 0) + count

    for f in range(N + 1):
        # initialize the first bin (index 0) to exactly f samples.
        clear(current_counts)
        current_counts[f][f][0] = comb(N, f) * (q[0] ** f)
        # n == 1: the single bin already holds all its items; close the cycle.
        if n == 1:
            readout(current_counts, f)
            continue

        for bin_idx in range(1, n):
            previous_counts, current_counts = current_counts, previous_counts
            clear(current_counts)
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
        readout(current_counts, f)

    # Pad with trailing zeros up to cdtv = 2*N (the maximum circular DTV).
    distribution = []
    for cdtv in range(0, 2 * N + 1):
        distribution.append((cdtv, result.get(cdtv, 0)))
    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Efficient (DP) exact circular DTV distribution for a "
                    "non-uniform null."
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
    distribution = calculate_exact_circular_dtv_distribution_nonuniform(args.N, q)

    lines = ["%d %d" % (cdtv, count) for cdtv, count in distribution]
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
