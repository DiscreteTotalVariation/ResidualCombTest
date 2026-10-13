"""Run-length testing of a bit source, in the exactly computable regime.

This is the application where every approximation can be switched off.

Null.  For a fair independent bitstream the successive run lengths are i.i.d.
with P(run = k) = 2^-k.  Pooling the tail at six bins gives
    q = (16, 8, 4, 2, 1, 1),   L = 32,
which is exactly rational with a small L, so the exact dynamic program of
dr_exact.py applies directly.  Nothing is estimated and nothing is fitted: the
null is known in closed form, unlike the image experiments where p0 is read off
the data.  Run-length histograms are also intrinsically ordered, which is what
DTV requires.

Alternative.  Two interleaved lanes.  A hardware generator that emits bits from
two alternating sources with slightly different flip probabilities,
    P(b_t != b_{t-1}) = 1/2 + (-1)^t eps,
is the bit-domain analogue of a two-way time-interleaved ADC.  It biases runs of
even length against runs of odd length, so the run-length histogram picks up a
period-2 comb on top of a steeply geometric baseline.  That is precisely an
alternative for the comb test under a non-uniform null, not under a uniform one.  The effect is second order: P(run = 1) moves by
about 2 eps^2, so eps = 0.10 is an l1 departure of only 0.07.

The alternative's run-length distribution is measured from a long simulation of
the generator rather than assumed, so the comb is produced by the source, not
imposed by hand.

Baselines.  The first version of this script reported only that the monobit
(frequency) test does not see the defect.  That is TRUE BY CONSTRUCTION -- the
generator keeps the bit frequency exactly balanced -- and a referee would rightly
call it a straw man.  Part B below therefore runs the NIST SP 800-22 tests that
actually target run structure (runs, serial m=2 and m=3, approximate entropy) on
the SAME bit windows from which the run-length histograms are taken, so every
statistic sees identical data and the comparison is at matched information.
The serial m=3 and approximate-entropy tests are the ones with a real chance:
two consecutive flips always straddle both lanes, so the 3-bit pattern
frequencies shift by order eps^2, the same order as the run-length comb.

Part A keeps the fixed-N multinomial design of the earlier version, extended with
a wider (eps, N) grid and the min-p combination over T(t), t in {0, 0.5, 1}.
"""

import argparse
from fractions import Fraction

import numpy as np

from dr_exact import calculate_exact_dr_distribution

NULL_WEIGHTS = (16, 8, 4, 2, 1, 1)
BINS = len(NULL_WEIGHTS)
EXPONENTS = (0.0, 0.5, 1.0)
HIST_KEYS = ["DTV", "CT", "T(1)", "min-p", "X2", "G"]
SERIAL_M = (2, 3, 4, 5, 6)          # NIST allows m <= floor(log2 n) - 2 = 8 at n = 2000
APEN_M = (2, 3, 4)                  # NIST allows m < floor(log2 n) - 5 = 5 at n = 2000
BIT_KEYS = (["mono", "runs", "dft", "lrun"] + [f"ser{m}" for m in SERIAL_M]
            + [f"ser{m}d" for m in SERIAL_M if m >= 3] + [f"apen{m}" for m in APEN_M])


# ---------------------------------------------------------------- generator --

def flip_probabilities(bits_count, eps):
    return 0.5 + eps * np.where(np.arange(bits_count) % 2 == 0, 1.0, -1.0)


def interleaved_runs(bits_count, eps, rng):
    """Run lengths of a two-lane generator, pooled at BINS (single long stream)."""
    flips = rng.random(bits_count) < flip_probabilities(bits_count, eps)
    bits = np.cumsum(flips) % 2
    boundaries = np.flatnonzero(np.diff(bits) != 0)
    lengths = np.diff(np.concatenate(([-1], boundaries, [len(bits) - 1])))
    lengths = lengths[1:-1]                      # drop the truncated end runs
    return np.clip(lengths, 1, BINS)


def bit_windows(count, width, eps, rng, random_phase=False):
    """`count` independent windows of `width` bits each, as a (count, width) array.

    With `random_phase` each window starts on a random lane, as a real capture
    window would; the default keeps the lane phase fixed at the window start.
    """
    probabilities = flip_probabilities(width + 1, eps)
    if random_phase:
        offset = rng.integers(0, 2, size=(count, 1))
        probabilities = probabilities[offset + np.arange(width)[None, :]]
    else:
        probabilities = probabilities[None, :width]
    flips = rng.random((count, width)) < probabilities
    start = rng.integers(0, 2, size=(count, 1))
    return (np.cumsum(flips, axis=1) + start) % 2


def run_length_histograms(bits):
    """Per-window run-length histograms pooled at BINS, end runs dropped."""
    count, width = bits.shape
    padded = np.empty((count, width + 2), dtype=bits.dtype)
    padded[:, 1:-1] = bits
    padded[:, 0] = 1 - bits[:, 0]
    padded[:, -1] = 1 - bits[:, -1]
    flat = padded.ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1])
    starts, ends = change[:-1] + 1, change[1:]
    stride = width + 2
    row_start, row_end = starts // stride, ends // stride
    interior = ((row_start == row_end)
                & (starts % stride >= 2)             # not the truncated first run
                & (ends % stride <= width - 1))      # not the truncated last run
    lengths = np.minimum(ends[interior] - starts[interior] + 1, BINS)
    rows = row_start[interior]
    histogram = np.bincount(rows * BINS + lengths - 1,
                            minlength=count * BINS).reshape(count, BINS)
    return histogram.astype(float)


# --------------------------------------------------------------- statistics --

def histogram_statistics(x, p0):
    """Histogram-domain statistics; N is taken per row so random N is fine."""
    N = x.sum(1, keepdims=True)
    expected = N * p0
    centered = x - expected
    scale = np.sqrt(N * p0 * (1.0 - p0))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / expected), 0.0)
    out = {
        "DTV": np.abs(np.diff(x, axis=1)).sum(1),
        "X2": (centered ** 2 / expected).sum(1),
        "G": 2.0 * terms.sum(1),
    }
    for t in EXPONENTS:
        out[f"T({t:g})"] = np.abs(np.diff(centered / scale ** t, axis=1)).sum(1)
    out["CT"] = out["T(0)"]           # d_r; T(1) is already keyed as "T(1)"
    return out


def pattern_counts(bits, m):
    """Counts of the 2^m overlapping m-bit patterns with cyclic extension."""
    count, width = bits.shape
    extended = np.concatenate([bits, bits[:, :m - 1]], axis=1) if m > 1 else bits
    index = np.zeros((count, width), dtype=np.int64)
    for j in range(m):
        index = (index << 1) | extended[:, j:j + width]
    offsets = (np.arange(count) << m)[:, None]
    return np.bincount((index + offsets).ravel(),
                       minlength=count << m).reshape(count, 1 << m).astype(float)


def psi_squared(bits, m):
    if m == 0:
        return np.zeros(bits.shape[0])
    width = bits.shape[1]
    return (2 ** m / width) * (pattern_counts(bits, m) ** 2).sum(1) - width


def longest_run_chisq(bits, block=8):
    """NIST SP 800-22, Sec. 2.4: longest run of ones in a block.

    This is the one test in the suite built directly on run structure, so it is
    the natural competitor for a run-length comb and must not be omitted.  With
    the lane phase held fixed it can BEAT the CT, because its block grid locks
    onto the defect's period; with the realistic random phase it cannot.
    """
    width = bits.shape[1] // block * block
    blocks = bits[:, :width].reshape(bits.shape[0], -1, block)
    best = np.zeros(blocks.shape[:2], dtype=np.int16)
    current = np.zeros(blocks.shape[:2], dtype=np.int16)
    for k in range(block):
        current = np.where(blocks[:, :, k] == 1, current + 1, 0)
        best = np.maximum(best, current)
    counts = np.stack([(best <= 1).sum(1), (best == 2).sum(1),
                       (best == 3).sum(1), (best >= 4).sum(1)], 1).astype(float)
    pi = np.array([0.2148, 0.3672, 0.2305, 0.1875])
    n_blocks = blocks.shape[1]
    return ((counts - n_blocks * pi) ** 2 / (n_blocks * pi)).sum(1)


def bit_statistics(bits):
    """NIST SP 800-22 style statistics, all vectorised over windows."""
    count, width = bits.shape
    ones = bits.sum(1)
    pi = ones / width
    flips = (bits[:, 1:] != bits[:, :-1]).sum(1)
    runs_count = 1 + flips
    with np.errstate(divide="ignore", invalid="ignore"):
        runs_z = np.abs(runs_count - 2 * width * pi * (1 - pi)) / (
            2 * np.sqrt(2 * width) * pi * (1 - pi))
    runs_z = np.where(np.isfinite(runs_z), runs_z, np.inf)
    psi = {m: psi_squared(bits, m) for m in range(0, max(SERIAL_M) + 1)}

    def phi(m):
        c = pattern_counts(bits, m) / width
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0)), 0.0).sum(1)

    out = {"mono": np.abs(2 * ones - width) / np.sqrt(width), "runs": runs_z,
           "lrun": longest_run_chisq(bits)}
    # discrete Fourier transform (spectral) test, NIST SP 800-22 Sec. 2.6: the
    # number of periodogram peaks below the 95% threshold, standardized; |d|
    threshold = np.sqrt(np.log(1.0 / 0.05) * width)
    below = np.empty(count)
    for start in range(0, count, 512):          # small blocks keep the FFT memory bounded
        block = bits[start:start + 512]
        spectrum = np.abs(np.fft.rfft(2.0 * block - 1.0, axis=1))[:, :width // 2]
        below[start:start + 512] = (spectrum < threshold).sum(1)
    out["dft"] = np.abs(below - 0.95 * width / 2) / np.sqrt(width * 0.95 * 0.05 / 4)
    # serial test: NIST reports both nabla psi^2_m (key ser{m}) and
    # nabla^2 psi^2_m (key ser{m}d) for each block length m
    for m in SERIAL_M:
        out[f"ser{m}"] = psi[m] - psi[m - 1]
        if m >= 3:
            out[f"ser{m}d"] = psi[m] - 2 * psi[m - 1] + psi[m - 2]
    # approximate entropy at block lengths m
    for m in APEN_M:
        out[f"apen{m}"] = 2 * width * (np.log(2) - (phi(m) - phi(m + 1)))
    return out


# --------------------------------------------------------------- inference --

def survival_transform(reference):
    ordered = np.sort(reference)
    size = ordered.size

    def transform(values):
        return 1.0 - np.searchsorted(ordered, values, side="left") / size
    return transform


def min_p(stats, transforms):
    return np.min([transforms[f"T({t:g})"](stats[f"T({t:g})"])
                   for t in EXPONENTS], axis=0)


def exact_critical(N, alpha):
    distribution = calculate_exact_dr_distribution(N, list(NULL_WEIGHTS))
    total = sum(NULL_WEIGHTS) ** N
    survival = total
    for value, count in distribution:
        survival -= count
        if Fraction(survival, total) <= alpha:
            return value, Fraction(survival, total)
    return distribution[-1][0], Fraction(0)


def chunked(total, chunk):
    done = 0
    while done < total:
        size = min(chunk, total - done)
        yield size
        done += size


# --------------------------------------------------------------------- main --

def part_a(args, rng, p0, alternatives, exact_cuts=None):
    print("PART A.  Fixed-N multinomial design, measured alternative law, "
          f"alpha={args.alpha}, MC-calibrated.")
    print("min-p = min over t in {0, 0.5, 1} of the p-value of T(t); its null "
          "calibration uses a SECOND independent null batch.\n")
    for N in args.N:
        x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
        x0b = rng.multinomial(N, p0, size=args.null_draws).astype(float)
        s0, s0b = histogram_statistics(x0, p0), histogram_statistics(x0b, p0)
        transforms = {f"T({t:g})": survival_transform(s0[f"T({t:g})"])
                      for t in EXPONENTS}
        cuts = {k: np.quantile(s0[k], 1.0 - args.alpha) for k in HIST_KEYS
                if k != "min-p"}
        cut_minp = np.quantile(min_p(s0b, transforms), args.alpha)
        exact_cut = (exact_cuts or {}).get(N)
        keys = HIST_KEYS + (["CT(exact)"] if exact_cut is not None else [])
        print(f"N = {N} runs   (mean count per bin {N / BINS:.1f})"
              + ("   [CT(exact) uses the EXACT critical value, no MC]" if exact_cut is not None else ""))
        print(f"{'eps':>6} {'l1':>8} " + " ".join(f"{k:>12}" if k == "CT(exact)" else f"{k:>7}" for k in keys))
        for eps in args.eps:
            x1 = rng.multinomial(N, alternatives[eps], size=args.alt_draws).astype(float)
            s1 = histogram_statistics(x1, p0)
            r = {k: float(np.mean(s1[k] > cuts[k])) for k in cuts}
            r["min-p"] = float(np.mean(min_p(s1, transforms) < cut_minp))
            if exact_cut is not None:
                r["CT(exact)"] = float(np.mean(s1["CT"] > exact_cut))
            tag = "  (size)" if eps == 0.0 else ""
            row = " ".join(f"{r[k]:12.3f}" if k == "CT(exact)" else f"{r[k]:7.3f}" for k in keys)
            print(f"{eps:>6.2f} {np.abs(alternatives[eps] - p0).sum():8.5f} {row}{tag}")
        print()


def collect(count, width, eps, rng, p0, chunk, random_phase=False):
    hist, bit = None, None
    for size in chunked(count, chunk):
        bits = bit_windows(size, width, eps, rng, random_phase)
        h = histogram_statistics(run_length_histograms(bits), p0)
        b = bit_statistics(bits)
        if hist is None:
            hist = {k: [v] for k, v in h.items()}
            bit = {k: [v] for k, v in b.items()}
        else:
            for k, v in h.items():
                hist[k].append(v)
            for k, v in b.items():
                bit[k].append(v)
    return ({k: np.concatenate(v) for k, v in hist.items()},
            {k: np.concatenate(v) for k, v in bit.items()})


def part_b(args, rng, p0):
    print("PART B.  Same-data design: every statistic is computed on the SAME "
          "windows of 2N bits.")
    print("Histogram statistics use the run-length histogram of the window (end "
          "runs dropped, so N is random with mean ~N);")
    print("bit statistics are NIST SP 800-22 style: mono = frequency, runs = runs test, dft = spectral (DFT) test, "
          "ser{m} = serial nabla psi^2_m and ser{m}d = serial nabla^2 psi^2_m for block "
          f"lengths m in {list(SERIAL_M)}, apen{{m}} = approximate entropy for m in {list(APEN_M)}; "
          "the letter quotes the best block length per test, which favors NIST.")
    print(f"alpha={args.alpha}, {args.window_null} null / {args.window_alt} "
          "alternative windows per cell, MC-calibrated.\n")
    keys = ["DTV", "CT", "min-p", "X2"] + BIT_KEYS
    for N in args.window_N:
        width = 2 * N
        h0, b0 = collect(args.window_null, width, 0.0, rng, p0, args.chunk, args.random_phase)
        h0b, _ = collect(args.window_null, width, 0.0, rng, p0, args.chunk, args.random_phase)
        transforms = {f"T({t:g})": survival_transform(h0[f"T({t:g})"])
                      for t in EXPONENTS}
        cuts = {k: np.quantile(h0[k], 1.0 - args.alpha) for k in ("DTV", "CT", "X2")}
        cuts.update({k: np.quantile(b0[k], 1.0 - args.alpha) for k in BIT_KEYS})
        cut_minp = np.quantile(min_p(h0b, transforms), args.alpha)
        print(f"windows of {width} bits  (~{N} runs)")
        print(f"{'eps':>6} " + " ".join(f"{k:>7}" for k in keys))
        for eps in args.eps:
            h1, b1 = collect(args.window_alt, width, eps, rng, p0, args.chunk, args.random_phase)
            r = {k: float(np.mean(h1[k] > cuts[k])) for k in ("DTV", "CT", "X2")}
            r.update({k: float(np.mean(b1[k] > cuts[k])) for k in BIT_KEYS})
            r["min-p"] = float(np.mean(min_p(h1, transforms) < cut_minp))
            tag = "  (size)" if eps == 0.0 else ""
            print(f"{eps:>6.2f} " + " ".join(f"{r[k]:7.3f}" for k in keys) + tag)
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, nargs="+",
                        default=[0.0, 0.02, 0.05, 0.07, 0.10, 0.15])
    parser.add_argument("--N", type=int, nargs="+",
                        default=[20, 50, 200, 1000, 5000])
    parser.add_argument("--bits", type=int, default=40_000_000,
                        help="bits simulated to measure the alternative")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=200000)
    parser.add_argument("--alt_draws", type=int, default=100000)
    parser.add_argument("--exact_N", type=int, nargs="+", default=[10, 20, 30, 100, 200],
                        help="N at which the exact L*d_r critical value is computed; "
                             "Part A then also reports CT power at that EXACT cut")
    parser.add_argument("--window_N", type=int, nargs="+", default=[200, 1000, 5000])
    parser.add_argument("--window_null", type=int, default=40000)
    parser.add_argument("--window_alt", type=int, default=20000)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--skip_b", action="store_true")
    parser.add_argument("--skip_a", action="store_true")
    parser.add_argument("--random_phase", action="store_true",
                        help="Part B: start each window on a random lane")
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    p0 = np.array(NULL_WEIGHTS, dtype=float) / sum(NULL_WEIGHTS)
    print("null: run lengths of a fair i.i.d. bitstream, "
          f"q={NULL_WEIGHTS}, L={sum(NULL_WEIGHTS)}, exactly rational")
    print(f"alternative: two-lane generator, P(flip) = 1/2 + (-1)^t eps, "
          f"measured from {args.bits:,} bits\n")

    print("measured run-length distribution of the generator")
    print(f"{'eps':>6} " + " ".join(f"{'k=' + str(k + 1):>8}" for k in range(BINS))
          + f" {'l1 from null':>13}")
    alternatives = {}
    for eps in args.eps:
        runs = interleaved_runs(args.bits, eps, rng)
        counts = np.bincount(runs, minlength=BINS + 1)[1:BINS + 1].astype(float)
        p1 = counts / counts.sum()
        alternatives[eps] = p1
        print(f"{eps:>6.2f} " + " ".join(f"{v:8.5f}" for v in p1)
              + f" {np.abs(p1 - p0).sum():13.5f}")
    print("(the null itself is " + " ".join(f"{v:.5f}" for v in p0) + ")\n")

    print("exact critical values for the CT (d_r) under the null "
          f"(alpha={args.alpha}), no Monte Carlo")
    print(f"{'N':>5} {'L*d_r crit':>13} {'exact size':>11}")
    exact_cuts = {}
    for N in args.exact_N:
        value, size = exact_critical(N, Fraction(args.alpha).limit_denominator(1000))
        exact_cuts[N] = value / sum(NULL_WEIGHTS)      # on the d_r scale
        print(f"{N:>5} {value:>13} {float(size):>11.4f}")
    print()

    if not args.skip_a:
        part_a(args, rng, p0, alternatives, exact_cuts)
    if args.random_phase:
        print("(Part B windows start on a RANDOM lane)\n")
    if not args.skip_b:
        part_b(args, rng, p0)


if __name__ == "__main__":
    main()
