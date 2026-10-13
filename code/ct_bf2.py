import argparse
from os import makedirs
from os.path import isfile, join

# Brute-force reference implementation of the exact DTV distribution using
# EXPLICIT (non-collapsed) enumeration, mirroring the naive algorithm from the
# ZDS-VU paper (Algorithm 1, uniform case with the q_i repetition loops dropped).
#
# Whereas ct_bf.py enumerates each distinct histogram once and weights it by its
# multinomial coefficient, this version materialises every one of the n**N bin
# assignments explicitly: it grows histograms one observation at a time, and a
# given histogram is produced once for each distinct way of reaching it. Each
# materialised histogram is then counted once. Both approaches yield identical
# distributions whose totals sum to n**N, but this one is far more memory- and
# time-intensive and is only practical for very small N and n.

DEFAULT_OUTPUT_DIR = "../data/exact_distributions_bf2/"


def enumerate_histograms_explicit(N, n):
    # Reproduce the paper's naive enumeration (uniform case): assign the sample
    # one observation at a time, branching into each of the n bins every step.
    if N == 0:
        return [[0] * n]

    # First observation: one histogram per bin.
    histograms = []
    for i in range(n):
        h = [0] * n
        h[i] = 1
        histograms.append(h)

    # Remaining observations 2..N: extend every histogram into every bin.
    for _ in range(2, N + 1):
        new_histograms = []
        for h in histograms:
            for i in range(n):
                h2 = list(h)
                h2[i] += 1
                new_histograms.append(h2)
        histograms = new_histograms

    return histograms


def compute_distribution_explicit(N, n):
    dtv_counts = dict()
    for h in enumerate_histograms_explicit(N, n):
        dtv = 0
        for i in range(1, n):
            dtv += abs(h[i] - h[i - 1])
        dtv_counts[dtv] = dtv_counts.get(dtv, 0) + 1

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
    # Mirror ct.py: save the distribution for every bin count from 1 to n.
    for m in range(1, n + 1):
        output_path = join(output_dir, "N_%d_n_%d.txt" % (N, m))
        if isfile(output_path):
            continue
        distribution = compute_distribution_explicit(N, m)
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
