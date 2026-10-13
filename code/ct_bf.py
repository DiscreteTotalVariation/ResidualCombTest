import argparse
from math import factorial
from os import makedirs
from os.path import isfile, join

# Brute-force reference implementation of the exact DTV distribution.
#
# Unlike ct.py, which uses a bin-by-bin dynamic-programming recurrence, this
# script literally enumerates every possible histogram (every ordered
# distribution of N samples into n bins) and accumulates the exact integer
# distribution from those. It is exponentially slower and is intended for
# verifying ct.py, not for production use.
#
# Each histogram (h_0, ..., h_{n-1}) with sum N contributes its multinomial
# coefficient N! / (h_0! ... h_{n-1}!) (the number of labelled samples giving
# that histogram) to the count of its DTV value, where
#   DTV = sum_{i=1}^{n-1} |h_i - h_{i-1}|.
# The totals therefore sum to n**N, matching ct.py.

DEFAULT_OUTPUT_DIR = "../data/exact_distributions_bf/"


def enumerate_histograms(total, parts):
    # Yield every ordered composition of `total` into `parts` non-negative parts.
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in enumerate_histograms(total - first, parts - 1):
            yield (first,) + rest


def compute_distribution_brute_force(N, n, factorials):
    dtv_counts = dict()
    for histogram in enumerate_histograms(N, n):
        dtv = 0
        for i in range(1, n):
            dtv += abs(histogram[i] - histogram[i - 1])
        weight = factorials[N]
        for h in histogram:
            weight //= factorials[h]
        dtv_counts[dtv] = dtv_counts.get(dtv, 0) + weight

    # Pad with trailing zeros up to dtv = 2*N, matching ct.py's output format.
    distribution = []
    for dtv in range(0, 2 * N + 1):
        distribution.append((dtv, dtv_counts.get(dtv, 0)))
    return distribution


def save_distribution(distribution, path):
    with open(path, "w", encoding="utf8") as fo:
        for a, b in distribution:
            fo.write(" ".join(map(str, [a, b])) + "\n")


def run_for_N(N, n, output_dir):
    # Precompute the factorials needed for the multinomial coefficients.
    factorials = [factorial(i) for i in range(N + 1)]

    # Mirror ct.py: save the distribution for every bin count from 1 to n.
    for m in range(1, n + 1):
        output_path = join(output_dir, "N_%d_n_%d.txt" % (N, m))
        if isfile(output_path):
            continue
        distribution = compute_distribution_brute_force(N, m, factorials)
        save_distribution(distribution, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-o", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-u", action="store_true", help="Run for all N from 1 to the given N")
    args = parser.parse_args()

    output_dir = args.output_dir
    makedirs(output_dir, exist_ok=True)

    N_values = range(1, args.N + 1) if args.u else [args.N]
    for N in N_values:
        run_for_N(N, args.n, output_dir)


if __name__ == "__main__":
    main()
