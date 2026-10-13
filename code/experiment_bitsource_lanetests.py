"""Lane-aware standard tests against the two-lane bit source (referee check).

Setting.  The two-lane generator of experiment_runlength_rng.py,
    P(b_t != b_{t-1}) = 1/2 + (-1)^t eps,
keeps the bit frequency and the MEAN flip rate at exactly 1/2.  That script
compares the comb test on the run-length histogram (CT, d_r; null
q = (16, 8, 4, 2, 1, 1), L = 32) with NIST SP 800-22 tests on the same windows.
A referee points out that standard tests which look at the lane structure
directly should see the defect much earlier, because the defect is FIRST order
in the per-lane flip rate, while the run-length comb is second order (P(run = 1)
moves by about 2 eps^2).  This script measures that, on the SAME generator
(imported, not copied), the same window widths and eps grid, the same
MC-calibration rule (cut = empirical 1 - alpha quantile of a null batch, reject
on strict >), and the random starting lane per window.

Tests, all computed on identical windows:

  CT      d_r of the window's run-length histogram, end runs dropped, exactly
          as in experiment_runlength_rng.py Part B (recomputed here, so the
          reproduction of the earlier random-lane numbers can be checked).
  pair    Knuth's serial test on NON-overlapping pairs (b_0 b_1)(b_2 b_3)...:
          Pearson chi-square of the four pair counts against n/4 each, 3 df.
          Under the null the n = floor(w/2) pairs are i.i.d. uniform on four
          cells, so chi^2_3 is its asymptotic law; both its asymptotic size and
          its MC-calibrated power are reported.  The within-pair flip always
          falls on the same lane, so P(01) + P(10) = 1/2 -+ eps; the chi-square
          is symmetric in the sign, hence does not depend on the starting lane.
  parity  D = (#flips at even positions) - (#flips at odd positions), flips
          b_j != b_{j-1} for j = 1..w-1 indexed within the window.  Under the
          null the w-1 flips are i.i.d. Bernoulli(1/2), so D = K - n_odd with
          K ~ Binomial(w-1, 1/2) EXACTLY.  When w-1 is odd the two parities
          have unequal counts and E0[D] = (n_even - n_odd)/2 = -1/2, so the
          two-sided statistic is the CENTRED |D - E0[D]| = |K - (w-1)/2|; the
          raw |D| would be lane-dependent by that half step.  The exact
          two-sided binomial cut is reported alongside the MC-calibrated one.

Lane invariance.  XOR-ing the window with 0101... complements every flip,
which is exactly the swap of the two lanes, and leaves a uniform start bit
uniform.  It permutes the four pair cells (00<->01, 10<->11), so the pair
chi-square is unchanged, and it maps D - E0[D] to its negative, so the centred
|D - E0[D]| is unchanged: both tests have the SAME law whichever lane comes
first, not merely the same law on average over a random start.

Standard errors.  "se" is the Monte Carlo standard error of a rejection rate
INCLUDING the calibration error of the MC cut: binomial variance of the
alternative batch plus the variance of the rate over a bootstrap of the null
batch (the cut re-estimated on each resample).  Rates at a fixed analytic cut
(chi^2_3, exact binomial) carry the binomial term only.

Seed.  33, the first value after the 13..32 block of code/SEEDS.md.  The
alphabetical 13 + i rule would put this script at 15 and shift every later
seed, so it is given the next free value instead; SEEDS.md still needs its row.
"""

import argparse
import os
import platform
import re
import sys

import numpy as np
import scipy
from scipy import stats

from experiment_runlength_rng import (NULL_WEIGHTS, bit_windows, chunked,
                                      histogram_statistics, run_length_histograms)

TESTS = ("CT", "pair", "parity")
EXISTING = ("experiment_runlength_rng_randomphase.txt",          # earlier random-lane
            "experiment_runlength_rng_randomphase_5000bits.txt")  # result files


# --------------------------------------------------------------- statistics --

def pair_chisq(bits):
    """Pearson chi-square of the non-overlapping pair counts (Knuth serial test)."""
    count, width = bits.shape
    n = width // 2
    code = 2 * bits[:, 0:2 * n:2] + bits[:, 1:2 * n:2]
    offsets = (np.arange(count) * 4)[:, None]
    counts = np.bincount((code + offsets).ravel(), minlength=4 * count).reshape(count, 4)
    expected = n / 4.0
    return ((counts - expected) ** 2).sum(1) / expected


def parity_split(bits):
    """D = flips at even positions minus flips at odd positions (signed), and E0[D]."""
    flips = (bits[:, 1:] != bits[:, :-1])            # flips[:, k] is position j = k + 1
    even = flips[:, 1::2].sum(1)                     # j = 2, 4, ...
    odd = flips[:, 0::2].sum(1)                      # j = 1, 3, ...
    n_even, n_odd = flips[:, 1::2].shape[1], flips[:, 0::2].shape[1]
    return (even - odd).astype(float), (n_even - n_odd) / 2.0


def window_statistics(count, width, eps, rng, p0, chunk):
    out = {k: [] for k in TESTS + ("D",)}
    for size in chunked(count, chunk):
        bits = bit_windows(size, width, eps, rng, random_phase=True)
        out["CT"].append(histogram_statistics(run_length_histograms(bits), p0)["CT"])
        out["pair"].append(pair_chisq(bits))
        d, null_mean = parity_split(bits)
        out["D"].append(d)
        out["parity"].append(np.abs(d - null_mean))
    return {k: np.concatenate(v) for k, v in out.items()}


# ---------------------------------------------------------------- inference --

def parity_null_law(width):
    """Support and pmf of |D - E0[D]| = |K - (w-1)/2|, K ~ Binomial(w-1, 1/2)."""
    m = width - 1
    k = np.arange(m + 1)
    return np.abs(k - m / 2.0), stats.binom.pmf(k, m, 0.5)


def exact_parity_cut(width, alpha):
    """Smallest c on the support with P(|D - E0 D| >= c) <= alpha, and that exact size."""
    values, pmf = parity_null_law(width)
    for c in np.unique(values):
        tail = pmf[values >= c].sum()
        if tail <= alpha:
            return c, tail
    return values.max() + 1, 0.0


def exact_parity_size_above(width, cut):
    """Exact null probability of the MC rule |D - E0 D| > cut."""
    values, pmf = parity_null_law(width)
    return pmf[values > cut].sum()


def bootstrap_cuts(null_values, alpha, replicates, rng):
    size = null_values.size
    return np.array([np.quantile(null_values[rng.integers(0, size, size)], 1.0 - alpha)
                     for _ in range(replicates)])


def rate_and_se(alt_values, cut, boot_cuts):
    """Rejection rate at the MC cut, with binomial + calibration standard error."""
    rate = float(np.mean(alt_values > cut))
    binom_var = rate * (1.0 - rate) / alt_values.size
    ordered = np.sort(alt_values)
    boot_rates = 1.0 - np.searchsorted(ordered, boot_cuts, side="right") / ordered.size
    return rate, float(np.sqrt(binom_var + boot_rates.var(ddof=1))), float(boot_rates.std(ddof=1))


def binomial_se(rate, n):
    return float(np.sqrt(rate * (1.0 - rate) / n))


# ------------------------------------------------- earlier results (check) --

def read_existing_ct(results_dir):
    """{(file, width): {eps: CT power}} parsed from the earlier random-lane files."""
    found = {}
    for name in EXISTING:
        path = os.path.join(results_dir, name)
        if not os.path.exists(path):
            continue
        width, columns, null_alt = None, None, None
        with open(path) as handle:
            for line in handle:
                m = re.search(r"(\d+) null / (\d+) alternative windows", line)
                if m:
                    null_alt = (int(m.group(1)), int(m.group(2)))
                m = re.match(r"windows of (\d+) bits", line)
                if m:
                    width, columns = int(m.group(1)), None
                    continue
                parts = line.split()
                if width is not None and parts and parts[0] == "eps":
                    columns = parts
                    continue
                if width is not None and columns and parts:
                    try:
                        eps = float(parts[0])
                    except ValueError:
                        continue
                    value = float(parts[columns.index("CT")])
                    found.setdefault((name, width), {"n": null_alt})[eps] = value
    return found


# --------------------------------------------------------------------- main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, nargs="+",
                        default=[0.0, 0.02, 0.05, 0.07, 0.10, 0.15])
    parser.add_argument("--widths", type=int, nargs="+", default=[400, 2000, 10000],
                        help="window widths in bits (Part B of experiment_runlength_rng.py "
                             "uses 2N for N in 200 1000 5000)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--window_null", type=int, default=200000)
    parser.add_argument("--window_alt", type=int, default=100000)
    parser.add_argument("--boot", type=int, default=200,
                        help="bootstrap replicates of the null batch for the calibration se")
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--results_dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    print("# provenance")
    print(f"# script : code/{os.path.basename(__file__)}")
    print(f"# command: python3 {' '.join([os.path.basename(sys.argv[0])] + sys.argv[1:])}")
    print(f"# seed   : {args.seed}   (numpy default_rng)")
    print(f"# python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    print("# generator, run-length histogram and CT (d_r) imported from experiment_runlength_rng.py")
    print()

    rng = np.random.default_rng(args.seed)
    p0 = np.array(NULL_WEIGHTS, dtype=float) / sum(NULL_WEIGHTS)
    chi_cut = stats.chi2.ppf(1.0 - args.alpha, 3)
    print("two-lane generator P(flip_t) = 1/2 + (-1)^t eps, each window starting on a RANDOM lane")
    print(f"alpha = {args.alpha};  {args.window_null} null windows (calibration) and "
          f"{args.window_alt} windows per eps row; eps = 0 row is a FRESH null batch (attained size)")
    print("CT     = d_r on the run-length histogram, q = "
          f"{NULL_WEIGHTS}, MC cut")
    print(f"pair   = non-overlapping pair chi-square (Knuth serial), 3 df; "
          f"'pair(chi2)' uses the asymptotic cut {chi_cut:.4f}")
    print("parity = |D - E0 D|, D = flips at even positions - flips at odd positions; "
          "'parity(exact)' uses the exact Binomial(w-1, 1/2) cut")
    print("se     = MC standard error; for MC-cut columns it includes the null-batch "
          "calibration error (bootstrap), for analytic cuts it is binomial only\n")

    results = {}
    for width in args.widths:
        null = window_statistics(args.window_null, width, 0.0, rng, p0, args.chunk)
        cuts = {k: np.quantile(null[k], 1.0 - args.alpha) for k in TESTS}
        boot = {k: bootstrap_cuts(null[k], args.alpha, args.boot, rng) for k in TESTS}
        c_par, size_par = exact_parity_cut(width, args.alpha)
        print(f"windows of {width} bits   (~{width // 2} runs, {width // 2} pairs, {width - 1} flips)")
        print(f"  MC cuts (reject on >): CT {cuts['CT']:.3f}, pair {cuts['pair']:.3f}, "
              f"parity {cuts['parity']:.2f} (exact size of that MC rule "
              f"{exact_parity_size_above(width, cuts['parity']):.4f})")
        print(f"  exact parity cut |D - E0 D| >= {c_par:g} has exact size {size_par:.4f}")
        print(f"  calibration batch: pair rejects above chi2_3 cut in "
              f"{np.mean(null['pair'] > chi_cut):.4f} of null windows; "
              f"mean D = {null['D'].mean():+.3f} (null {-0.5 * ((width - 1) % 2):+.1f}), var D = {null['D'].var():.1f} "
              f"(null (w-1)/4 = {(width - 1) / 4:.1f})")
        header = (f"  {'eps':>5} " + " ".join(f"{k:>14}" for k in
                  ("CT", "pair", "parity", "pair(chi2)", "parity(exact)")) + "   [rate (se)]")
        print(header)
        for eps in args.eps:
            alt = window_statistics(args.window_alt, width, eps, rng, p0, args.chunk)
            row = {}
            for k in TESTS:
                row[k] = rate_and_se(alt[k], cuts[k], boot[k])
            r = float(np.mean(alt["pair"] > chi_cut))
            row["pair(chi2)"] = (r, binomial_se(r, args.window_alt), 0.0)
            r = float(np.mean(alt["parity"] >= c_par))
            row["parity(exact)"] = (r, binomial_se(r, args.window_alt), 0.0)
            results[(width, eps)] = row
            tag = "  (size)" if eps == 0.0 else ""
            print(f"  {eps:5.2f} " + " ".join(f"{row[k][0]:7.4f}({row[k][1]:.4f})" for k in
                  ("CT", "pair", "parity", "pair(chi2)", "parity(exact)")) + tag)
        print()

    # ------------------------------------------------ reproduction check --
    print("REPRODUCTION CHECK: CT power here vs the earlier random-lane runs of "
          "experiment_runlength_rng.py --random_phase (seed 24, independent stream).")
    print("z = difference / sqrt(se_here^2 + se_earlier^2); se_earlier = binomial on its "
          "alternative batch + calibration se scaled from this run by its null batch size.")
    existing = read_existing_ct(args.results_dir)
    if not existing:
        print("  (earlier result files not found; check skipped)\n")
    worst = 0.0
    for (name, width), table in sorted(existing.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        n_null, n_alt = table["n"]
        print(f"  {name}, windows of {width} bits ({n_null} null / {n_alt} alt):")
        if width not in args.widths:
            print("    (width not run here)")
            continue
        print(f"    {'eps':>5} {'earlier':>8} {'here':>8} {'diff':>8} {'z':>6}")
        for eps, earlier in sorted((e, v) for e, v in table.items() if e != "n"):
            if (width, eps) not in results:
                continue
            here, se_here, calib_here = results[(width, eps)]["CT"]
            calib_earlier = calib_here * np.sqrt(args.window_null / n_null)
            se_earlier = np.sqrt(earlier * (1 - earlier) / n_alt + calib_earlier ** 2)
            se = np.sqrt(se_here ** 2 + se_earlier ** 2)
            if se > 0:
                z = (here - earlier) / se
                worst = max(worst, abs(z))
                ztext = f"{z:+6.2f}"
            else:
                ztext = f"{'n/a':>6}"             # both at 1.000 (or 0): no MC spread
            print(f"    {eps:5.2f} {earlier:8.3f} {here:8.4f} {here - earlier:+8.4f} {ztext}")
    if existing:
        print(f"  largest |z| = {worst:.2f}  (earlier values are rounded to 3 decimals, "
              "which adds up to 0.0005 not counted in the se)")
    print()

    # ------------------------------------------------------- comparison --
    print("WINDOW NEEDED, rough: power at the MC cut, CT vs the lane-aware tests")
    print(f"  {'width':>6} {'eps':>5} {'CT':>7} {'pair':>7} {'parity':>7}")
    for width in args.widths:
        for eps in args.eps:
            if eps == 0.0:
                continue
            row = results[(width, eps)]
            print(f"  {width:>6} {eps:5.2f} " + " ".join(f"{row[k][0]:7.3f}" for k in TESTS))


if __name__ == "__main__":
    main()
