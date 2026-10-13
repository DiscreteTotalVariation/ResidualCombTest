"""Run the comb test (CT) with a non-uniform null r on one histogram.

This is the entry point for using the test, as opposed to reproducing the
Letter.  Give it the observed counts and the null, and it reports the
statistic d_r = ||x - N r||_V (the DTV of the centered counts, the centered
total variation), a p-value, a decision, and the collapse index of that null.

    python3 comb_test.py --counts 12 5 9 4 11 --weights 1 2 1 2 1
    python3 comb_test.py --counts 12 5 9 4 11 --probs 0.1 0.3 0.1 0.3 0.2
    python3 comb_test.py --counts-file hist.txt --weights 16 8 4 2 1 1

The null must be KNOWN, not estimated from the same data or from a short
calibration capture: see GLOSSARY.md.  The expected histogram minimizes the
statistic, so the test is upper-tailed: large values signal a departure.

Calibration is chosen automatically:

  exact   the exact null of Section II-A (dynamic program), no Monte Carlo
          anywhere.  It needs a rational null r = q/L with integer weights q,
          and is used when N <= --max-exact-N, the number of bins
          n <= --max-exact-n, the weight total L <= --max-exact-L, and the
          estimated run time is within --max-exact-seconds.  The run time grows
          as n * N^4 * c, where c <= min(L, 2^(n-1)) is the number of residue
          classes of S = L*d_r mod L that the null can reach; it is printed
          before every exact run.  Probabilities given to many decimals have a
          huge L and are never put on the lattice: they take the gamma route,
          computed in floating point on the normalized probabilities.
  gamma   moment-matched to null draws of d_r, as in Section IV-A.  Used beyond
          the exact regime.  Its attained size is 0.039 to 0.054 at alpha = 0.05
          over the Letter's nulls, and its cut lies above the exact one at
          alpha <= 0.01.  Its p-value is a fitted tail, not an exact
          probability.

Use --minp for Tippett's combination over T(0), T(0.5), T(1) when you do not
know whether the departure scales the probabilities or shifts them
(Section IV-B).  min-p is always calibrated by Monte Carlo (two batches of
--draws null histograms), never exactly.  Its cost in power depends on the
null: on the Letter's synthetic Block 1 (four nulls, both comb kinds, N = 500)
its worst shortfall behind the best member is 0.03, against 0.22 for
committing to T(0) alone; on the image nulls of Table II (multiplicative
code-width combs) it trails the CT = T(0) by up to 0.07.  Choose between the
CT and min-p before looking at the data.
"""

import argparse
import sys

import numpy as np

from du_exact import prepare_integer_weights
from dr_exact import calculate_exact_dr_distribution

# Seconds per unit of n * N^4 * c for the exact dynamic program, measured on
# this repository's nulls (0.8e-9 to 7.6e-9 s; the estimate is a rough guide,
# and it overestimates when c is large, because the reachable values are then
# sparse).
SECONDS_PER_UNIT = 3e-9


def integer_statistic(counts, q):
    """S = L * d_r, an integer: L*(x_i - x_{i-1}) - N*(q_i - q_{i-1}), summed."""
    L, N = sum(q), sum(counts)
    return sum(abs(L * (counts[i] - counts[i - 1]) - N * (q[i] - q[i - 1]))
               for i in range(1, len(counts)))


def float_statistic(counts, p):
    """d_r = ||x - N p||_V in floating point, for any (also irrational) null p."""
    x = np.asarray(counts, float)
    p = np.asarray(p, float) / np.sum(p)
    return float(np.abs(np.diff(x - x.sum() * p)).sum())


def residue_count(N, q):
    """c: residue classes of S mod L reachable by the sign patterns of Eq. (S).

    Every term of S is congruent to +-N(q_i - q_{i-1}) mod L, so S mod L lies in
    the set of signed sums of those n-1 residues: at most min(L, 2^(n-1))
    classes.  This is the cost factor of the exact null (Section II-A)."""
    L = sum(q)
    classes = {0}
    for i in range(1, len(q)):
        t = (N * (q[i] - q[i - 1])) % L
        classes = {(v + t) % L for v in classes} | {(v - t) % L for v in classes}
    return len(classes)


def exact_tail(N, q, s_obs, alpha):
    """(p-value, cut, attained size) from the exact null of S, no Monte Carlo."""
    dist = calculate_exact_dr_distribution(N, q)
    total = sum(c for _, c in dist)
    at_or_above = sum(c for v, c in dist if v >= s_obs)
    # Section II-A: reject when S exceeds the smallest s with Pr[S > s] <= alpha.
    cut, size, above = None, None, 0
    for v, c in sorted(dist, reverse=True):          # walk down from the top
        if c == 0:
            continue
        if above / total <= alpha:
            cut, size = v, above / total             # v is the largest such s so far
        above += c
    return at_or_above / total, cut, size


def family(x, p, N, t):
    """T(t) = sum |y_i - y_{i-1}| with y_i = e_i / (N p_i (1-p_i))^(t/2), e = x - N p."""
    r = x - N * p
    y = r / (N * p * (1.0 - p)) ** (t / 2.0) if t else r
    return np.abs(np.diff(y, axis=-1)).sum(-1)


def gamma_tail(N, p, d_obs, alpha, draws, rng):
    """(p-value, cut, attained size on an independent batch), all on the d_r scale.

    `p` is the null as floats (weights are normalized here); nothing is put on
    an integer lattice, so any null works, including irrational ones."""
    p = np.asarray(p, float) / np.sum(p)
    fit = family(rng.multinomial(N, p, size=draws).astype(float), p, N, 0)
    mean, var = float(fit.mean()), float(fit.var())
    if var == 0.0:
        # d_r is constant under this null (tiny N or a near-degenerate null):
        # there is no tail to fit.  Report the trivial p-value rather than NaN.
        return (1.0 if d_obs <= mean else 0.0), mean, 0.0
    shape, scale = mean ** 2 / var, var / mean
    from scipy.stats import gamma as gamma_dist
    cut = float(gamma_dist.ppf(1.0 - alpha, shape, scale=scale))
    test = family(rng.multinomial(N, p, size=draws).astype(float), p, N, 0)
    return float(gamma_dist.sf(float(d_obs), shape, scale=scale)), cut, float((test > cut).mean())


def collapse_index(q, N, draws, rng):
    """corr_H0(||x||_V, Lambda(x)): how far the uncentered DTV d_u has collapsed
    into the signed sum Lambda on this null (Section III); 1 = fully."""
    p = np.asarray(q, float) / sum(q)
    x = rng.multinomial(N, p, size=draws).astype(float)
    dtv = np.abs(np.diff(x, axis=1)).sum(1)
    lam = (np.sign(np.diff(p)) * np.diff(x, axis=1)).sum(1)
    if lam.std() == 0 or dtv.std() == 0:
        return 0.0
    return float(np.corrcoef(dtv, lam)[0, 1])


def min_p_calibrate(q, N, alpha, draws, rng, return_null=False):
    """Tippett's combination over t in {0, 0.5, 1}, calibrated on two batches.

    Returns (pmin, cut): `pmin(samples)` maps histograms (rows) to the minimum of
    the three empirical upper-tail p-values, and `cut` is its alpha-quantile
    under the null.  The minimum is NOT itself a p-value (Section IV-B), so
    batch 1 supplies the per-statistic p-value maps and an independent batch 2
    supplies the null distribution of the minimum.  Small min-p is evidence, so
    the test rejects when pmin(x) <= cut, a LOWER-tail rule.  With
    `return_null=True` the batch-2 minima are returned as a third element, from
    which a calibrated p-value is read (see `min_p`).
    """
    p = np.asarray(q, float) / sum(q)
    ts = (0.0, 0.5, 1.0)
    batch1 = rng.multinomial(N, p, size=draws).astype(float)
    batch2 = rng.multinomial(N, p, size=draws).astype(float)
    null1 = {t: np.sort(family(batch1, p, N, t)) for t in ts}
    K = float(draws)

    def pmin(sample):
        # upper-tail p-value of each member, then the minimum over members
        per_t = [1.0 - np.searchsorted(null1[t], family(sample, p, N, t), "left") / K
                 for t in ts]
        return np.min(per_t, axis=0)

    # "lower" keeps the cut at an attained value: min-p is heavily discrete, and
    # an interpolated quantile plus "<=" would sweep in a whole atom and inflate
    # the size (measured 0.070 against 0.05 before this was fixed).
    null2 = pmin(batch2)
    cut = float(np.quantile(null2, alpha, method="lower"))
    return (pmin, cut, null2) if return_null else (pmin, cut)


def min_p(counts, q, N, alpha, draws, rng):
    """(observed minimum, cut, calibrated p-value).

    The observed minimum of the three member p-values is NOT a p-value.  The
    calibrated p-value is the Monte Carlo probability, under the null, of a
    minimum at or below the observed one: (1 + #{batch 2 <= obs}) / (draws + 1).
    The decision is calibrated p-value <= alpha."""
    pmin, cut, null2 = min_p_calibrate(q, N, alpha, draws, rng, return_null=True)
    obs = float(pmin(np.asarray(counts, float)[None, :])[0])
    pval = (1.0 + float(np.sum(null2 <= obs))) / (len(null2) + 1.0)
    return obs, cut, pval


def fmt_null(p, limit=12):
    """Short text for a float null; long ones are elided."""
    vals = [f"{v:.4g}" for v in p]
    if len(vals) > limit:
        vals = vals[:6] + ["..."] + vals[-2:]
    return "(" + ", ".join(vals) + ")"


def main():
    ap = argparse.ArgumentParser(
        description="Comb test (CT) with null r on one histogram: the statistic is the "
                    "DTV of the centered counts, d_r = ||x - N r||_V.")
    c = ap.add_mutually_exclusive_group(required=True)
    c.add_argument("--counts", type=int, nargs="+", help="observed bin counts, in bin order")
    c.add_argument("--counts-file", help="text file of whitespace-separated counts, in bin order")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", type=int, nargs="+",
                   help="integer null weights q, r = q/L (exact null possible)")
    g.add_argument("--probs", type=float, nargs="+",
                   help="null probabilities r; put on an integer lattice only when that "
                        "lattice is small enough for the exact null, else used as floats")
    ap.add_argument("--alpha", type=float, default=0.05, help="level, strictly between 0 and 1")
    ap.add_argument("--draws", type=int, default=50000,
                    help="null draws when simulating (gamma fit, collapse index, min-p)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max-exact-N", type=int, default=200, metavar="N_MAX",
                    help="above this N the gamma is used instead of the exact null")
    ap.add_argument("--max-exact-n", type=int, default=12, metavar="BINS_MAX",
                    help="above this many bins the gamma is used (the recurrence slows with n)")
    ap.add_argument("--max-exact-L", type=int, default=1000, metavar="L_MAX",
                    help="above this weight total L the gamma is used (probabilities given "
                         "to many decimals are not small-denominator rationals)")
    ap.add_argument("--max-exact-seconds", type=float, default=1800.0, metavar="SECONDS",
                    help="above this estimated run time (from n * N^4 * c, c the residue "
                         "count) the gamma is used; raise it to force a long exact run")
    ap.add_argument("--gamma", action="store_true", help="force the gamma route")
    ap.add_argument("--minp", action="store_true",
                    help="also run the min-p combination over T(0), T(0.5), T(1) "
                         "(always Monte Carlo)")
    a = ap.parse_args()

    if a.counts_file:
        try:
            x = [int(v) for v in open(a.counts_file).read().split()]
        except (OSError, ValueError) as e:
            sys.exit(f"error: cannot read integer counts from {a.counts_file}: {e}")
    else:
        x = a.counts
    if not 0.0 < a.alpha < 1.0:
        sys.exit("error: --alpha must be strictly between 0 and 1")
    if a.probs:
        if min(a.probs) <= 0:
            sys.exit("error: every null probability must be positive")
        total = sum(a.probs)
        if abs(total - 1.0) > 1e-6:
            print(f"note: the probabilities sum to {total:.6g}, not 1; they are renormalized")
        nul = a.probs
    else:
        if min(a.weights) <= 0:
            sys.exit("error: every null weight must be positive")
        nul = a.weights
    if len(x) != len(nul):
        sys.exit(f"error: {len(x)} counts but {len(nul)} null bins")
    if len(x) < 2:
        sys.exit("error: the statistic needs at least two bins")
    if min(x) < 0:
        sys.exit("error: counts cannot be negative")
    if sum(x) == 0:
        sys.exit("error: all counts are zero, there is nothing to test")
    N, n = sum(x), len(x)
    p = np.asarray(nul, float) / float(sum(nul))   # the null as floats, always defined
    rng = np.random.default_rng(a.seed)

    # Route.  The integer lattice q (r = q/L) is built only when the exact null
    # is still possible; probabilities with many decimals would otherwise give
    # an L with hundreds of digits, which floating-point code cannot take.
    too_big = []
    if N > a.max_exact_N:
        too_big.append(f"N = {N} > {a.max_exact_N}")
    if n > a.max_exact_n:
        too_big.append(f"n = {n} bins > {a.max_exact_n}")
    q = None
    if not a.gamma and not too_big:
        q = list(a.weights) if a.weights else prepare_integer_weights([v / total for v in a.probs])
        L = sum(q)
        if L > a.max_exact_L:
            shown = str(L) if L < 10 ** 12 else f"a {len(str(L))}-digit integer"
            too_big.append(f"L = {shown} > {a.max_exact_L}"
                           + (" (for an exact null, round r to integer --weights"
                              " q_i ~ L r_i with a small L)" if a.probs else ""))
            q = None
    elif a.weights:
        L = sum(a.weights)
        if L > a.max_exact_L:
            too_big.append(f"L = {L} > {a.max_exact_L}")
    c_res = secs = None
    if q is not None:
        c_res = residue_count(N, q)
        secs = SECONDS_PER_UNIT * n * N ** 4 * c_res
        if secs > a.max_exact_seconds:
            too_big.append(f"estimated exact run time ~{secs:.0f} s > --max-exact-seconds "
                           f"{a.max_exact_seconds:.0f} (n*N^4*c with c = {c_res} residue classes)")
            q = None
    exact_ok = q is not None

    print(f"comb test (CT) with null r:  n = {n} bins, N = {N} counts, alpha = {a.alpha}")
    if exact_ok:
        s = integer_statistic(x, q)
        print(f"  null r = q/L,  q = {tuple(q)},  L = {L}")
        print(f"  statistic  d_r = ||x - N r||_V = {s / L:.6g}   (lattice value S = L*d_r = {s})")
        if secs > 10:
            print(f"  exact null: c = {c_res} residue classes (at most min(L, 2^(n-1)) = "
                  f"{min(L, 2 ** (n - 1))});\n    cost ~ n*N^4*c, estimated ~{secs:.0f} s "
                  f"(rough; long runs can also need several GB of memory)", flush=True)
        pval, cut, size = exact_tail(N, q, s, a.alpha)
        print(f"  calibration: EXACT (Section II-A, dynamic program, no Monte Carlo)")
        print(f"  reject when S > {cut}, i.e. d_r > {cut / L:.6g}   "
              f"(attained size {size:.4f}, not {a.alpha})")
        print(f"  p-value  Pr[S >= {s}] = {pval:.4f}   (exact)")
    else:
        d = float_statistic(x, p)
        if a.weights:
            print(f"  null r = q/L,  q = {tuple(a.weights)},  L = {sum(a.weights)}")
        else:
            print(f"  null r = {fmt_null(p)}   (used in floating point)")
        print(f"  statistic  d_r = ||x - N r||_V = {d:.6g}")
        why = "forced" if a.gamma else "beyond the exact regime: " + "; ".join(too_big)
        pval, cut, size = gamma_tail(N, p, d, a.alpha, a.draws, rng)
        print(f"  calibration: GAMMA moment-matched to {a.draws} null draws (Section IV-A)")
        print(f"    ({why})")
        print(f"  reject when d_r > {cut:.6g}   (size {size:.4f} on an independent batch)")
        print(f"  p-value (gamma fit, approximate) ~ {pval:.4f}")
    print(f"  decision: {'REJECT the null' if pval <= a.alpha else 'do not reject'}")

    # Monte Carlo below runs on q when the exact route was taken (bit-identical
    # to earlier versions there), else on the null exactly as given.
    mc_null = q if exact_ok else nul
    idx_draws = min(a.draws, 20000)
    idx = collapse_index(mc_null, N, idx_draws, rng)
    se = (1.0 - idx ** 2) / np.sqrt(idx_draws)          # s.e. of a correlation
    print(f"\n  collapse index of this null at N = {N}: {idx:.2f}   (MC, {idx_draws} draws, "
          f"s.e. ~{max(se, 0.001):.3f})")
    print("    How far the uncentered DTV d_u = ||x||_V would have collapsed, on this null,")
    print("    into a fixed signed sum of the counts (Section III): near 0 it would keep both")
    print("    comb signs, near 1 it would detect only one.  There is no threshold.  The CT")
    print("    (d_r) is centered and does not collapse: its powers against the two comb signs")
    print("    differ only slightly.  The index matters only if d_u were used instead.")

    if a.minp:
        obs, mcut, mp = min_p(x, mc_null, N, a.alpha, a.draws, rng)
        print(f"\n  min-p combination over T(0), T(0.5), T(1) (Section IV-B), always Monte Carlo")
        print(f"  (two batches of {a.draws} null draws; p-value resolution 1/{a.draws + 1}):")
        print(f"    minimum of the three member p-values {obs:.4f}   (NOT a p-value)")
        print(f"    calibrated p-value ~ {mp:.4f}  ->  "
              f"{'REJECT the null' if mp <= a.alpha else 'do not reject'}")
        print("    Report the CT decision above or this one, chosen before looking at the data;")
        print("    reporting whichever rejects inflates the size.  min-p suits a departure that")
        print("    may scale or shift the probabilities.  Its worst shortfall behind the best")
        print("    member is 0.03 on the Letter's synthetic Block 1 only; on the image nulls of")
        print("    Table II it trails the CT by up to 0.07.")


if __name__ == "__main__":
    main()
