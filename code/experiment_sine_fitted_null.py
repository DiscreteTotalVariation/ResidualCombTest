"""IEEE 1241 sine-wave histogram test with the arcsine null FITTED from the record.

`experiment_sine_histogram_adc.py` builds the arcsine code-density null from the
TRUE sine amplitude and offset.  In IEEE Std 1241 practice neither is known: both
are estimated from the same record.  This script repeats the N = 10^4, eps = 0.05
SAR-mismatch row with the null estimated record by record, and asks what that does
to the size and power of the comb test (CT, statistic d_r) and of Pearson's
X^2.

Sampling assumption.  Every sample sees an independent, uniformly distributed sine
phase (asynchronous / random-phase sampling), so the N codes are i.i.d. and the
code counts are MULTINOMIAL(N, p).  Coherent sampling (an integer number of
periods with relatively prime cycle and sample counts, which IEEE 1241 also
describes) spreads the phases evenly instead; the counts are then far less
variable than multinomial and nothing below applies to it unchanged.

Estimator (IEEE 1241 histogram method, reconstructed from the standard's closed
form; written from the published equations as we understand them, not copied from
the standard's text).  With S samples, H[k] the count in code k and H_c the
cumulative histogram, the standard estimates the transition levels as
    T[k] = C - A cos(pi H_c[k-1] / S),   k = 1 .. n-1,
which is the inverse of the arcsine CDF.  Offset C and amplitude A are scale
factors the histogram cannot fix on its own; the standard fixes them from the two
END codes by requiring the first and last transitions to sit at their nominal
(ideal) levels T[1] = -1 + 2/n, T[n-1] = 1 - 2/n:
    C1 = cos(pi H[0] / S),     C2 = cos(pi H[n-1] / S),
    A^ = (T[n-1] - T[1]) / (C1 + C2),
    C^ = (T[n-1] C1 + T[1] C2) / (C1 + C2).
This is what is implemented (`ieee1241_estimate`).  It uses only the two end-code
counts, it is exactly consistent under the ideal quantiser, and the fitted null
reproduces the observed end-code counts exactly (p^_0 = H[0]/S, p^_{n-1} =
H[n-1]/S), so two degrees of freedom are spent.  Under the SAR-mismatch
alternative the first and last transitions are NOT at their nominal levels (each
moves by eps LSB), so the fit is biased there: it absorbs part of the defect,
which is what would happen in practice.

Procedures compared, all at alpha = 0.05:
  true   : null from the TRUE (A, C), cut by MC from that null (the existing
           experiment; must reproduce its numbers within MC error);
  fitted, true-null cut : null fitted per record, statistic computed against it,
           compared with the cut calibrated for the TRUE-parameter null (what a
           practitioner with a precomputed cut does).  For the CT both the MC cut
           and the moment-matched gamma cut; for X^2 the true-null MC cut and the
           asymptotic chi2(n-1) and chi2(n-3) cuts;
  fitted, oracle cut : the cut recalibrated for the FITTED statistic by MC at the
           true parameters (not available in practice; shows what the correct
           cut is);
  fitted, bootstrap  : per-record parametric bootstrap: B records drawn from the
           fitted null, each re-fitted, the statistic recomputed; reject when the
           bootstrap p-value (1 + #{T* >= T}) / (B + 1) <= alpha.

Seed.  SEEDS.md numbers seeds 13 + alphabetical index; inserting this script into
that list would renumber `experiment_sine_histogram_adc.py` and every later
script, so it takes the next free seed, 33, outside the numbering.
"""

import argparse
import platform
import sys
import time

import numpy as np
import scipy
from scipy import stats

from experiment_sine_histogram_adc import (code_probabilities, ideal_edges,
                                           mismatched_edges)


def ieee1241_estimate(x, n):
    """IEEE 1241 histogram estimates (A^, C^) from end-code counts; x is (R, n)."""
    S = x.sum(1)
    c1 = np.cos(np.pi * x[:, 0] / S)
    c2 = np.cos(np.pi * x[:, n - 1] / S)
    t_first, t_last = -1.0 + 2.0 / n, 1.0 - 2.0 / n
    amplitude = (t_last - t_first) / (c1 + c2)
    offset = (t_last * c1 + t_first * c2) / (c1 + c2)
    return amplitude, offset


def fitted_nulls(x, edges):
    """Arcsine code probabilities from the per-record IEEE 1241 estimates, (R, n)."""
    n = x.shape[1]
    amplitude, offset = ieee1241_estimate(x, n)
    inner = edges[None, 1:-1] - offset[:, None]
    cdf = 0.5 + np.arcsin(np.clip(inner / amplitude[:, None], -1.0, 1.0)) / np.pi
    ones = np.ones((x.shape[0], 1))
    return np.diff(np.concatenate((0.0 * ones, cdf, ones), axis=1), axis=1)


def dr_and_x2(x, p, N):
    """d_r = sum |diff(x - N p)| and Pearson X^2; p is (n,) or (R, n)."""
    expected = N * np.broadcast_to(p, x.shape)
    centered = x - expected
    dr = np.abs(np.diff(centered, axis=1)).sum(1)
    x2 = (centered ** 2 / np.maximum(expected, 1e-300)).sum(1)
    return dr, x2


def rate(flags):
    flags = np.asarray(flags, dtype=float)
    r = flags.mean()
    return r, np.sqrt(r * (1.0 - r) / flags.size)


def fmt(pair):
    return f"{pair[0]:.3f} ({pair[1]:.3f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=1.02)
    parser.add_argument("--offset", type=float, default=0.0,
                        help="true sine offset in full-scale units")
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=40000,
                        help="null draws per calibration batch (as the parent script)")
    parser.add_argument("--records", type=int, default=100000,
                        help="records per condition for the non-bootstrap procedures")
    parser.add_argument("--boot_records", type=int, default=10000,
                        help="records per condition for the parametric bootstrap")
    parser.add_argument("--B", type=int, default=999, help="bootstrap replicates")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    n, N, A, C, alpha = args.n, args.N, args.amplitude, args.offset, args.alpha
    edges = ideal_edges(n)
    p_true = code_probabilities(edges, A, offset=C)

    print("# experiment_sine_fitted_null.py: " + " ".join(sys.argv[1:] or ["(defaults)"]))
    print(f"# python {platform.python_version()}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}; seed {args.seed}")
    print(f"IEEE 1241 sine-wave histogram test with a FITTED arcsine null: n={n} codes, "
          f"true amplitude {A} FS, true offset {C} FS, N={N}, SAR comb eps=+-{args.eps}, "
          f"alpha={alpha}")
    print("Sampling: random phase (asynchronous), so counts are multinomial(N, p).")
    print("Estimator: IEEE 1241 histogram closed form from the end-code counts, "
          "first/last transitions pinned at nominal; see the docstring.\n")

    # sanity: the estimator recovers (A, C) on a very long clean record
    big = rng.multinomial(10 ** 9, p_true, size=1).astype(float)
    a_big, c_big = ieee1241_estimate(big, n)
    print(f"estimator check, one record of 1e9 samples: A^={a_big[0]:.5f} "
          f"(true {A}), C^={c_big[0]:+.6f} (true {C})")

    # ---- calibration on the TRUE-parameter null (what a precomputed cut is) ----
    cal = rng.multinomial(N, p_true, size=args.null_draws).astype(float)
    cal_t, cal_x2 = dr_and_x2(cal, p_true, N)
    cut_t = np.quantile(cal_t, 1.0 - alpha)
    cut_x2 = np.quantile(cal_x2, 1.0 - alpha)
    gam = rng.multinomial(N, p_true, size=args.null_draws).astype(float)
    gam_t, _ = dr_and_x2(gam, p_true, N)
    shape, scale = gam_t.mean() ** 2 / gam_t.var(), gam_t.var() / gam_t.mean()
    cut_gamma = stats.gamma.isf(alpha, shape, scale=scale)
    cut_x2_asy1 = stats.chi2.isf(alpha, n - 1)
    cut_x2_asy3 = stats.chi2.isf(alpha, n - 3)

    # ---- oracle cut for the FITTED statistic (MC at the true parameters) ----
    orc = rng.multinomial(N, p_true, size=args.null_draws).astype(float)
    orc_t, orc_x2 = dr_and_x2(orc, fitted_nulls(orc, edges), N)
    cut_t_fit = np.quantile(orc_t, 1.0 - alpha)
    cut_x2_fit = np.quantile(orc_x2, 1.0 - alpha)
    a_h0, c_h0 = ieee1241_estimate(orc, n)

    print(f"estimates under H0 at N={N}: A^ mean {a_h0.mean():.5f} sd {a_h0.std():.5f}; "
          f"C^ mean {c_h0.mean():+.6f} sd {c_h0.std():.6f} (FS; 1 LSB = {2 / n:.5f})")
    print(f"cuts, true-parameter null:  CT MC {cut_t:.1f}, CT gamma {cut_gamma:.1f}, "
          f"X2 MC {cut_x2:.2f}, X2 chi2({n - 1}) {cut_x2_asy1:.2f}, "
          f"chi2({n - 3}) {cut_x2_asy3:.2f}")
    print(f"cuts, fitted statistic (oracle MC at true parameters): CT {cut_t_fit:.1f}, "
          f"X2 {cut_x2_fit:.2f}")
    print(f"null mean of the statistic: CT true {cal_t.mean():.1f} vs fitted "
          f"{orc_t.mean():.1f};  X2 true {cal_x2.mean():.2f} vs fitted {orc_x2.mean():.2f}\n")

    conditions = [("H0 (size)", p_true),
                  (f"comb eps=+{args.eps}", code_probabilities(mismatched_edges(n, args.eps), A, offset=C)),
                  (f"comb eps=-{args.eps}", code_probabilities(mismatched_edges(n, -args.eps), A, offset=C))]

    columns = ["CT true/MC", "CT true/gam", "X2 true/MC",
               "CT fit/trueMC", "CT fit/trueGam", "X2 fit/trueMC",
               "X2 fit/chi2(n-1)", "X2 fit/chi2(n-3)",
               "CT fit/oracle", "X2 fit/oracle",
               "CT fit/boot", "X2 fit/boot"]
    results = {}
    for label, p1 in conditions:
        x = rng.multinomial(N, p1, size=args.records).astype(float)
        t_true, x2_true = dr_and_x2(x, p_true, N)
        p_hat = fitted_nulls(x, edges)
        t_fit, x2_fit = dr_and_x2(x, p_hat, N)
        r = {"CT true/MC": rate(t_true > cut_t),
             "CT true/gam": rate(t_true > cut_gamma),
             "X2 true/MC": rate(x2_true > cut_x2),
             "CT fit/trueMC": rate(t_fit > cut_t),
             "CT fit/trueGam": rate(t_fit > cut_gamma),
             "X2 fit/trueMC": rate(x2_fit > cut_x2),
             "X2 fit/chi2(n-1)": rate(x2_fit > cut_x2_asy1),
             "X2 fit/chi2(n-3)": rate(x2_fit > cut_x2_asy3),
             "CT fit/oracle": rate(t_fit > cut_t_fit),
             "X2 fit/oracle": rate(x2_fit > cut_x2_fit)}
        a_hat, c_hat = ieee1241_estimate(x, n)

        # per-record parametric bootstrap on an independent set of records
        xb = rng.multinomial(N, p1, size=args.boot_records).astype(float)
        pb_hat = fitted_nulls(xb, edges)
        tb_obs, x2b_obs = dr_and_x2(xb, pb_hat, N)
        reject_t = np.empty(args.boot_records, dtype=bool)
        reject_x2 = np.empty(args.boot_records, dtype=bool)
        for i in range(args.boot_records):
            star = rng.multinomial(N, pb_hat[i], size=args.B).astype(float)
            t_star, x2_star = dr_and_x2(star, fitted_nulls(star, edges), N)
            reject_t[i] = (1 + np.sum(t_star >= tb_obs[i])) / (args.B + 1) <= alpha
            reject_x2[i] = (1 + np.sum(x2_star >= x2b_obs[i])) / (args.B + 1) <= alpha
        r["CT fit/boot"] = rate(reject_t)
        r["X2 fit/boot"] = rate(reject_x2)
        results[label] = r
        print(f"{label}: A^ mean {a_hat.mean():.5f}, C^ mean {c_hat.mean():+.6f}  "
              f"[{time.time() - started:.0f} s]")

    print(f"\nrejection rate (MC standard error); {args.records} records per condition, "
          f"bootstrap columns {args.boot_records} records x B={args.B}")
    width = max(len(c) for c in columns)
    labels = [lab for lab, _ in conditions]
    print(f"{'procedure':<{width}}  " + "  ".join(f"{lab:>16}" for lab in labels)
          + f"  {'worse sign':>16}")
    for col in columns:
        cells = [results[lab][col] for lab in labels]
        worse = min(cells[1:], key=lambda c: c[0])
        print(f"{col:<{width}}  " + "  ".join(f"{fmt(c):>16}" for c in cells)
              + f"  {fmt(worse):>16}")

    print("\nreference, experiment_sine_histogram_adc.txt at N=10000, eps=+-0.05: "
          "size CT 0.050, X2 0.048; power CT 0.772/0.772 (MC), 0.770/0.771 "
          "(gamma), X2 0.508/0.509")
    print(f"runtime {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
