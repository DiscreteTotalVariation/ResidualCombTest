import argparse
import sys
from os import makedirs
from os.path import dirname

import numpy as np

# Monte Carlo estimation of the DTV distribution for a NON-UNIFORM null
# distribution. This is the Monte Carlo analogue of du_exact.py, following the
# multinomial-sampling style of ../zds_vu/code/monte_carlo_chi_squared.py:
# it draws N observations from the discrete distribution ps a large number of
# times, computes the discrete total variation of each simulated histogram, and
# tallies the empirical distribution.
#
# DTV = sum_{i=1}^{n-1} |h_i - h_{i-1}|.
#
# The output uses the same "dtv count" format as du_exact.py, where count is the
# number of simulations giving that DTV (so counts sum to num_simulations).
# Dividing by num_simulations gives probabilities comparable to the exact
# distribution from du_exact.py (whose counts sum to L**N).


def monte_carlo_dtv_distribution(N, ps, num_simulations=10000, seed=13):
    n = len(ps)

    # Normalize probabilities.
    ps_arr = np.array(ps, dtype=float)
    ps_arr = ps_arr / ps_arr.sum()

    rng = np.random.RandomState(seed)

    # Draw N observations from the multinomial distribution num_simulations times.
    simulated_counts = rng.multinomial(N, ps_arr, size=num_simulations)

    # Discrete total variation of each simulated histogram.
    if n >= 2:
        dtv = np.abs(np.diff(simulated_counts, axis=1)).sum(axis=1)
    else:
        dtv = np.zeros(num_simulations, dtype=int)

    # Empirical counts, padded with trailing zeros up to dtv = 2*N.
    counts = np.bincount(dtv, minlength=2 * N + 1)
    distribution = [(d, int(counts[d])) for d in range(0, 2 * N + 1)]
    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo estimation of the DTV distribution for a "
                    "non-uniform null."
    )
    parser.add_argument("-N", type=int, required=True, help="Sample size")
    parser.add_argument("--probs", type=float, nargs="+", required=True,
                        help="Per-bin probabilities (or proportional weights); "
                             "the number of bins n is inferred from this list")
    parser.add_argument("--num_simulations", "-n", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--seed", "-s", type=int, default=13,
                        help="Random seed (default: 13)")
    parser.add_argument("-o", dest="output", default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    distribution = monte_carlo_dtv_distribution(
        N=args.N, ps=args.probs,
        num_simulations=args.num_simulations, seed=args.seed,
    )

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
