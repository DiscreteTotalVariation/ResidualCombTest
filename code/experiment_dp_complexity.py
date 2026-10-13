"""Empirical complexity of the exact dynamic program of the centered DTV d_r.

A reviewer of the parent paper asked for the stated complexity to be justified
rather than asserted.  The naive claim is that d_r costs L times the uncentered
DTV d_u, because L*d_r ranges over O(L*N) values instead of O(N).  That is
wrong, and this script is the evidence.

Each increment of the statistic is |L*(x_i - x_{i-1}) - N*(q_i - q_{i-1})|, whose
value is congruent to +-N*(q_i - q_{i-1}) modulo L (the sign depends on which
way the absolute value resolves).  The accumulated statistic mod L therefore
takes at most c(q) <= min(L, 2^(n-1)) values -- a number fixed by q, independent
of N and of L -- so the reachable values lie on c(q) translates of the lattice
L*Z inside the O(L*N) range, and their number is O(c(q)*N), not O(L*N).
Measured c is 1 to 16 for the nulls used in the paper at N <= 100, and is
    larger at odd N than even; it is NOT a function of q and N mod L alone.  The dictionary-based
DP visits only reachable values, so the true cost is

  reachable statistic values   O(c(q) * N),  a small multiple of d_u's,
  states                       O(N^3),
  transitions                  O(c(q) * n * N^4) time,

i.e. the d_u cost times a q-dependent constant, with L entering only through
big-integer digit length (log L).  The wide range shows up only in the
zero-padded output, not in the work done.  (An earlier version of this text said
"a fixed residue class"; the +- makes that false, and the corrected count is what
the 'classes' measurement below reports.)

This measures the support size and the running time and reports the fitted
log-log slopes, so the exponents can be read off rather than trusted.
"""

import argparse
import time

import numpy as np

from du_exact import calculate_exact_dtv_distribution_nonuniform
from dr_exact import calculate_exact_dr_distribution


def measure(function, *arguments):
    start = time.perf_counter()
    distribution = function(*arguments)
    elapsed = time.perf_counter() - start
    support = sum(1 for _, count in distribution if count)
    return elapsed, support, len(distribution)


def residue_classes(distribution, L):
    """Number of distinct values of S mod L over the support: the c(q) above."""
    return len({value % L for value, count in distribution if count})


def slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(x, y, 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=[10, 15, 20, 30, 45])
    parser.add_argument("--classes_N", type=int, nargs="+",
                        default=[10, 15, 20, 25, 40, 100])  # odd N included: c is larger there
    args = parser.parse_args()

    print("scaling in N, weights q=(1,2,1,2,1), n=5, L=7")
    print(f"{'N':>5} {'support':>9} {'d_r s':>11} {'d_u s':>10} "
          f"{'ratio':>7}")
    q = [1, 2, 1, 2, 1]
    sizes, times, du_times = [], [], []
    for N in args.N:
        elapsed, support, _ = measure(calculate_exact_dr_distribution,
                                      N, q)
        du_elapsed, _, _ = measure(calculate_exact_dtv_distribution_nonuniform,
                                    N, q)
        sizes.append(support)
        times.append(elapsed)
        du_times.append(du_elapsed)
        print(f"{N:>5} {support:>9} {elapsed:>11.3f} {du_elapsed:>10.3f} "
              f"{elapsed / max(du_elapsed, 1e-9):>7.1f}")
    print(f"fitted slopes in N:  support {slope(args.N, sizes):.2f} "
          f"(predicted 1),  time {slope(args.N, times):.2f} (predicted 4), "
          f"d_u time {slope(args.N, du_times):.2f} (predicted 4)\n")

    print("scaling in L at fixed N=15, n=5, weights q=(1,k,1,k,1)")
    print("the 'range' column is the O(L*N) span of L*d_r; 'support' is what "
          "the DP actually visits")
    print(f"{'L':>5} {'k':>4} {'range':>9} {'support':>9} {'density':>8} "
          f"{'seconds':>9}")
    weights, supports, elapsed_list = [], [], []
    for k in (1, 2, 4, 8, 16, 32):
        q = [1, k, 1, k, 1]
        total = sum(q)
        elapsed, support, span = measure(calculate_exact_dr_distribution,
                                         15, q)
        weights.append(total)
        supports.append(support)
        elapsed_list.append(elapsed)
        print(f"{total:>5} {k:>4} {span:>9} {support:>9} {support / span:>8.3f} "
              f"{elapsed:>9.3f}")
    print(f"fitted slopes in L:  support {slope(weights, supports):.2f} "
          f"(predicted 0),  time {slope(weights, elapsed_list):.2f} "
          f"(predicted 0)\n")

    print("residue classes c = number of distinct values of S mod L on "
          "the support, for the letter's nulls; bound min(L, 2^(n-1))")
    print(f"{'null':>28} {'L':>5} {'bound':>6} " + " ".join(f"N={N:<4}" for N in args.classes_N))
    for name, q in (("uniform 1:1:1:1:1", [1, 1, 1, 1, 1]), ("comb 1:2:1:2:1", [1, 2, 1, 2, 1]),
                    ("tent 1:2:3:4:3:2:1", [1, 2, 3, 4, 3, 2, 1]),
                    ("ramp 1:2:3:4:5:6", [1, 2, 3, 4, 5, 6]),
                    ("run-length 16:8:4:2:1:1", [16, 8, 4, 2, 1, 1])):
        L = sum(q)
        classes = [residue_classes(calculate_exact_dr_distribution(N, q), L)
                   for N in args.classes_N]
        print(f"{name:>28} {L:>5} {min(L, 2 ** (len(q) - 1)):>6} "
              + " ".join(f"{c:<6}" for c in classes))


if __name__ == "__main__":
    main()
