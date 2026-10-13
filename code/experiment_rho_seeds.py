"""How stable is the collapse-index sweep across seeds?

The sweep draws part of its null family at random, so its summary statistics are
themselves random.  An earlier version of the Letter quoted the rank correlation
to two decimals and quoted the index at which the uncentered d_u first reaches
alpha; both are extrema or point estimates over a random family and move with
the seed.
This script measures how much, so the Letter can quote a range instead of a
number it cannot reproduce.  The Spearman correlation is that of the index with
d_u's worse-sign power, so it is negative (index up, power down), as in the Letter.
"""

import argparse
import subprocess
import sys
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[22, 501, 502, 503, 504])
    args = parser.parse_args()
    print("Collapse-index sweep repeated at several seeds.  The null family is")
    print("partly random, so these summaries are random too.\n")
    print(f"{'seed':>6} {'nulls':>6} {'Spearman':>9} {'CT worse-sign range':>22}")
    spearman, lows, highs = [], [], []
    for seed in args.seeds:
        out = subprocess.run([sys.executable, "experiment_rho_sweep.py", "--seed", str(seed)],
                             capture_output=True, text=True).stdout
        rho = re.search(r"Spearman (-?[\d.]+)", out)
        rng = re.search(r"over all (\d+) nulls: ([\d.]+) to ([\d.]+)", out)
        if not (rho and rng):
            print(f"{seed:>6}  could not parse"); continue
        r = float(rho.group(1)); lo, hi = float(rng.group(2)), float(rng.group(3))
        spearman.append(r); lows.append(lo); highs.append(hi)
        print(f"{seed:>6} {int(rng.group(1)):>6} {r:>9.3f} {lo:>11.3f} to {hi:<8.3f}", flush=True)
    if spearman:
        print(f"\nSpearman over seeds: {min(spearman):.2f} to {max(spearman):.2f}")
        print(f"CT worse-sign floor over seeds: {min(lows):.3f}")
        print("\nQuote the RANGE, not a two-decimal point estimate, and do not quote the")
        print("index at which d_u first reaches alpha: that is a minimum over a")
        print("random family and is not reproducible.")


if __name__ == "__main__":
    main()
