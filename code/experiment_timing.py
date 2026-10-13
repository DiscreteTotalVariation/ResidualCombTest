"""Wall-clock cost of the exact dynamic program, and the reach of exactness.

The Letter quotes two timings in its Cost paragraph.  Both were previously
recorded in a results file that no script produced, which makes them
unreproducible; this script produces them.

Timings are machine-dependent by nature, so this results file will NOT
reproduce byte for byte (nor will the timing columns of
experiment_dp_complexity.txt).  The support sizes and the residue counts beside
them will.
"""

import argparse
import statistics
import time

from du_exact import calculate_exact_dtv_distribution_nonuniform as du_law
from dr_exact import calculate_exact_dr_distribution as dr_law

NULL = (16, 8, 4, 2, 1, 1)          # the run-length null of Section V-C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--fair", action="store_true",
                        help="time the uncentered d_u with the SAME sparse program (dr_exact.py, raw=True) "
                             "instead of the dense reference du_exact.py; the dense reference "
                             "re-zeroes an (N+1)x(N+1)x(2N+2) buffer per bin, which makes the "
                             "exact DP of d_r look cheaper than it is at large N")
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed runs per cell; the median is reported, because a "
                             "single sub-second run is not reproducible to two digits")
    args = parser.parse_args()
    L = sum(NULL)
    print("Wall-clock cost of the exact DP of d_r, and the d_u DP for reference"
          + (" (SAME sparse program, increment |m-k|: like for like)." if args.fair
             else " (dense reference du_exact.py; NOT like for like, see --fair)."))
    print(f"null q={NULL}, L={L}.  Timings are machine-dependent; the Letter quotes")
    print(f"only the ratio, to one digit.  Median of {args.repeats} runs per cell.\n")
    print(f"{'N':>6} {'d_r s':>11} {'d_u s':>8} {'ratio':>6} {'support':>8} {'c':>4}")
    for N in args.N:
        dr_t, du_t = [], []
        for _ in range(args.repeats):
            t = time.perf_counter(); law = dr_law(N, list(NULL)); dr_t.append(time.perf_counter() - t)
            t = time.perf_counter()
            if args.fair:
                dr_law(N, list(NULL), raw=True)
            else:
                du_law(N, list(NULL))
            du_t.append(time.perf_counter() - t)
        rs, ws = statistics.median(dr_t), statistics.median(du_t)
        support = sum(1 for _, c in law if c)
        classes = len({v % L for v, c in law if c})
        print(f"{N:>6} {rs:>11.1f} {ws:>8.1f} {rs / max(ws, 1e-9):>6.1f} "
              f"{support:>8} {classes:>4}", flush=True)
    print("\nThe residue count c is NOT a function of q and N mod L alone: it depends on")
    print("which sign patterns are reachable, and is larger for odd N than even.")


if __name__ == "__main__":
    main()
