import argparse
import sys
from fractions import Fraction as F
from functools import reduce
from math import factorial, gcd
from os import makedirs
from os.path import dirname

# Brute-force exact DTV distribution for a NON-UNIFORM null distribution, using
# COLLAPSED (non-explicit) enumeration. This is the collapsed counterpart of
# du_exact_bf2.py: instead of materialising all L**N histograms, it enumerates each
# distinct histogram once and weights it by the number of times the explicit
# ZDS-VU enumeration would produce it.
#
# For a histogram (h_0, ..., h_{n-1}) with sum N and integer bin weights q_i,
# that count is the multinomial coefficient times the product of weights:
#   N! / (h_0! ... h_{n-1}!) * prod_i q_i**h_i,
# (multinomial = number of orderings of the sample; q_i**h_i = choices among the
# q_i repeated slots of bin i). Summed over all histograms this gives L**N,
# where L = sum_i q_i, matching du_exact_bf2.py exactly.
#
# When all q_i = 1 (uniform case) this reduces to ct_bf.py, with counts summing
# to n**N. The DTV statistic is DTV = sum_{i=1}^{n-1} |h_i - h_{i-1}|.


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


def enumerate_histograms(total, parts):
    # Yield every ordered composition of `total` into `parts` non-negative parts.
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in enumerate_histograms(total - first, parts - 1):
            yield (first,) + rest


def compute_distribution_nonuniform(N, q):
    n = len(q)
    factorials = [factorial(i) for i in range(N + 1)]

    dtv_counts = dict()
    for histogram in enumerate_histograms(N, n):
        dtv = 0
        for i in range(1, n):
            dtv += abs(histogram[i] - histogram[i - 1])
        weight = factorials[N]
        for i in range(n):
            weight //= factorials[histogram[i]]
            weight *= q[i] ** histogram[i]
        dtv_counts[dtv] = dtv_counts.get(dtv, 0) + weight

    # Pad with trailing zeros up to dtv = 2*N, matching ct.py's output format.
    distribution = []
    for dtv in range(0, 2 * N + 1):
        distribution.append((dtv, dtv_counts.get(dtv, 0)))
    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Brute-force exact DTV distribution for a non-uniform null, "
                    "using collapsed enumeration."
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
