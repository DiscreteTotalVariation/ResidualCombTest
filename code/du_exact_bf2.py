import argparse
import sys
from fractions import Fraction as F
from functools import reduce
from math import gcd
from os import makedirs
from os.path import dirname

# Brute-force exact DTV distribution for a NON-UNIFORM null distribution, using
# the explicit (non-collapsed) enumeration of the ZDS-VU paper (Algorithm 1,
# "Naive calculation ... for non-uniform distribution with integer weights"),
# applied to the DTV statistic instead of chi-squared.
#
# Mechanism (mirrors the paper):
#   1. The bin probabilities p_i = a_i/b_i are turned into integer weights
#      q_i = p_i * L, where L = lcm(b_1, ..., b_n), via prepare_integer_weights.
#   2. The sample is assigned one observation at a time, branching into every
#      bin. The non-uniformity is handled by the shaded "for j in 1..q(i)" loops:
#      each assignment to bin i is materialised q_i times, so more probable bins
#      contribute proportionally more histograms.
#   3. This produces L**N histograms in total; the DTV of each is tallied once.
#
# When all q_i are equal (uniform case) this reduces exactly to ct_bf2.py, with
# counts summing to n**N. In general the counts sum to L**N. Like the other
# brute-force scripts it materialises every histogram, so it is only practical
# for very small N and small L.


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


def enumerate_histograms_nonuniform(N, q):
    # Explicit enumeration with the paper's q_i repetition loops included.
    n = len(q)
    if N == 0:
        # No observations: a single all-zero histogram (L**0 = 1).
        return [[0] * n]

    # First observation: histogram per bin, each repeated q_i times.
    histograms = []
    for i in range(n):
        h = [0] * n
        h[i] = 1
        for _ in range(q[i]):
            histograms.append(h)

    # Remaining observations 2..N: extend every histogram into every bin,
    # again repeating each bin-i assignment q_i times.
    for _ in range(2, N + 1):
        new_histograms = []
        for h in histograms:
            for i in range(n):
                h2 = list(h)
                h2[i] += 1
                for _ in range(q[i]):
                    new_histograms.append(h2)
        histograms = new_histograms

    return histograms


def compute_distribution_nonuniform(N, q):
    dtv_counts = dict()
    for h in enumerate_histograms_nonuniform(N, q):
        dtv = 0
        for i in range(1, len(h)):
            dtv += abs(h[i] - h[i - 1])
        dtv_counts[dtv] = dtv_counts.get(dtv, 0) + 1

    # Pad with trailing zeros up to dtv = 2*N, matching ct.py's output format.
    distribution = []
    for dtv in range(0, 2 * N + 1):
        distribution.append((dtv, dtv_counts.get(dtv, 0)))
    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Brute-force exact DTV distribution for a non-uniform null, "
                    "using the explicit ZDS-VU enumeration."
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
    distribution = compute_distribution_nonuniform(args.N, q)

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
