"""Comb detection on real image code histograms against a NON-UNIFORM null.

The parent Comb Test paper (Section IV-C3) had to pass every real image through a
dithered rank transform first, because CT can only test uniformity: the rank
transform manufactures a flat ideal code distribution so that the test has a null
it can handle.  That step is a workaround, not part of the measurement chain of
any real converter.

Here the rank transform is removed.  The signal driving the quantiser is the raw
image intensity, so the ideal code distribution is the image's own histogram --
steeply non-uniform, and exactly rational by construction (p_i = c_i / M, with
c_i the number of source pixels landing in code i and M the pixel count).  No
estimation is involved and no distributional assumption is made.

Measurement chain per image:
  1. analog surrogate  v = intensity + U(0,1)  on [0, 256),
  2. ideal quantiser: n equal codes            -> null p0 = c/M,
  3. defective quantiser: alternating widths W_k proportional to 1 + (-1)^k eps
     (the LSB capacitor-mismatch model of the parent paper) -> alternative p1,
  4. draw N pixels and test the n-bin code histogram.

eps = 0 reproduces the ideal codes exactly, so that row is a size check.

The SIGN of eps is the PHASE of the defect: eps > 0 makes the even codes wide,
eps < 0 makes the odd codes wide.  Both are the same physical fault and are
equally likely, and the comb test of the parent paper is advertised as detecting
such a departure without knowing its phase.  Running both signs is therefore not
a robustness check but the main measurement: a statistic whose power depends on
the sign of eps is not a comb test on this null.

Every critical value is Monte-Carlo calibrated from p0, so all tests are compared
at the same actual size.  The asymptotic chi-square rejection rate is reported
separately, because that is what a practitioner would actually run and its size
distortion at these (n, N) is part of the story.

Statistics
  minp     Tippett's min-p over T(0), T(0.5), T(1) (Section IV-B), calibrated on a
           second independent null batch; rejects at or below its cut
  DTV      uncentered DTV d_u = sum |x_i - x_{i-1}|, upper tail: the naive way to
           point the parent uniform-null CT at a non-uniform null.
  DTV2     the same statistic, two-sided.
  CT       d_r = sum |e_i - e_{i-1}|,  e_i = x_i - N p_i (the centered counts).
  T(1)     sum |z_i - z_{i-1}|,  z_i = e_i / sqrt(N p_i (1 - p_i)).
  X2       Pearson chi-square (MC-calibrated).
  G        likelihood-ratio G-test (MC-calibrated).
  AC       |sum_i (-1)^i e_i|, the locally most powerful oracle for a known
           period-2 comb.
  vN       von Neumann ratio sum (e_i - e_{i-1})^2 / sum e_i^2, the L2 analogue.
  RUN      number of runs in sign(x_i - N p_i).
  X2p      Pearson chi-square after pooling adjacent bins to expected count >= 5
           (groups fixed by the null), MC-calibrated like X2.
  X2asy    Pearson chi-square against its asymptotic chi^2_{n-1} null.
"""

import argparse

import numpy as np
import skimage.color as skcolor
import skimage.data as skdata
from scipy import stats

# Images whose code histogram is strictly positive in every bin at n = 64, so
# that the non-uniform null has no zero-probability cells.
IMAGES = ["camera", "coins", "moon", "page", "cell"]
ORDER = ["DTV", "DTV2", "CT", "T(1)", "minp", "X2", "X2p", "G", "AC", "vN", "RUN", "X2asy"]
MINP_T = (0.0, 0.5, 1.0)   # the members of Tippett's min-p combination, Section IV-B


def load_gray_uint8(name):
    image = np.asarray(getattr(skdata, name)())
    if image.ndim == 3:
        image = skcolor.rgb2gray(image) * 255.0
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def quantise(v, n, eps):
    """Quantise v in [0, 256) into n codes with widths 1 + (-1)^k eps."""
    return np.clip(np.searchsorted(code_edges(n, eps), v, side="right") - 1, 0, n - 1)


def code_edges(n, eps):
    widths = 1.0 + eps * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    edges = 256.0 * np.concatenate(([0.0], np.cumsum(widths / widths.sum())))
    edges[-1] = 256.0
    return edges


def code_probabilities(image, n, eps):
    """Exact code law of the analog surrogate v = intensity + U(0, 1): a pixel of
    intensity I spreads unit mass uniformly over [I, I+1), which the code edges
    cut; p_k = sum_I c_I |[I, I+1) meet [e_k, e_{k+1})|.  For eps = 0 and n
    dividing 256 this is exactly the image's own n-code histogram."""
    mass = np.bincount(image.ravel(), minlength=256).astype(float)
    mass /= mass.sum()
    edges = code_edges(n, eps)
    intensity = np.arange(256, dtype=float)
    low = np.maximum(intensity[:, None], edges[None, :-1])
    high = np.minimum(intensity[:, None] + 1.0, edges[None, 1:])
    return mass @ np.clip(high - low, 0.0, None)


def pooling_edges(expected, minimum=5.0):
    """Start indices of adjacent bin groups whose expected count is >= minimum
    (the textbook remedy for sparse cells), from the null alone."""
    edges, accumulated = [0], 0.0
    for j, e in enumerate(expected):
        accumulated += e
        if accumulated >= minimum and j < len(expected) - 1:
            edges.append(j + 1)
            accumulated = 0.0
    if len(edges) > 1 and expected[edges[-1]:].sum() < minimum:
        edges.pop()
    return np.array(edges)


def statistics(x, p0, N):
    expected = N * p0
    centered = x - expected
    edges = pooling_edges(expected)
    pooled_centered = np.add.reduceat(x, edges, axis=1) - np.add.reduceat(expected, edges)
    pooled_expected = np.add.reduceat(expected, edges)
    z = centered / np.sqrt(N * p0 * (1.0 - p0))
    y_half = centered / (N * p0 * (1.0 - p0)) ** 0.25     # T(0.5) of Section IV-B
    signs = np.where(np.arange(x.shape[1]) % 2 == 0, 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / expected), 0.0)
    # sign runs of the centered counts, ignoring exact zeros (a zero continues the
    # previous sign); the earlier version counted (+,0,+) as one change.
    centered_signs = np.sign(centered)
    index = np.where(centered_signs != 0, np.arange(centered_signs.shape[1])[None, :], -1)
    last = np.maximum.accumulate(index, axis=1)
    filled = np.where(last >= 0, np.take_along_axis(centered_signs, np.maximum(last, 0), axis=1), 0)
    changes = (filled[:, 1:] != filled[:, :-1]) & (filled[:, 1:] != 0) & (filled[:, :-1] != 0)
    return {
        "DTV": np.abs(np.diff(x, axis=1)).sum(1),
        "DTV2": np.abs(np.diff(x, axis=1)).sum(1),
        "CT": np.abs(np.diff(centered, axis=1)).sum(1),
        "T(1)": np.abs(np.diff(z, axis=1)).sum(1),
        "T(0.5)": np.abs(np.diff(y_half, axis=1)).sum(1),
        "X2": (centered ** 2 / expected).sum(1),
        "X2p": (pooled_centered ** 2 / pooled_expected).sum(1),
        "G": 2.0 * terms.sum(1),
        "AC": np.abs(centered @ signs),
        "vN": (np.diff(centered, axis=1) ** 2).sum(1) / (centered ** 2).sum(1),
        "RUN": changes.sum(1).astype(float),
    }


MEMBER = {0.0: "CT", 0.5: "T(0.5)", 1.0: "T(1)"}


def min_p_calibration(null1, null2, alpha):
    """Tippett's min-p over T(0), T(0.5), T(1), calibrated as in Section IV-B.

    null1 supplies the per-member empirical p-value maps; null2, an INDEPENDENT
    null batch, supplies the null distribution of the minimum, whose alpha-quantile
    is the cut.  The minimum is not itself a p-value, so it is rejected when it is
    AT OR BELOW the cut (small = evidence).  Returns (pmin function, cut)."""
    sorted1 = {t: np.sort(null1[MEMBER[t]]) for t in MINP_T}
    K = float(len(sorted1[0.0]))

    def pmin(stats_dict):
        per_t = [1.0 - np.searchsorted(sorted1[t], stats_dict[MEMBER[t]], "left") / K
                 for t in MINP_T]
        return np.min(per_t, axis=0)

    cut = float(np.quantile(pmin(null2), alpha, method="lower"))
    return pmin, cut


def rejection_rates(null_stats, alt_stats, alpha):
    rates = {}
    for key in null_stats:
        if key == "T(0.5)":
            continue
        null_values, alt_values = null_stats[key], alt_stats[key]
        if key == "DTV2":
            lo, hi = np.quantile(null_values, [alpha / 2.0, 1.0 - alpha / 2.0])
            rates[key] = float(np.mean((alt_values > hi) | (alt_values < lo)))
        else:
            threshold = np.quantile(null_values, 1.0 - alpha)
            rates[key] = float(np.mean(alt_values > threshold))
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--N", type=int, nargs="+",
                        default=[500, 2000, 10000, 50000])
    parser.add_argument("--eps", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=40000)
    parser.add_argument("--alt_draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--images", type=str, nargs="+", default=None,
                        help="subset of images to pool over (default: all five); "
                             "pass a single image for per-image results")
    args = parser.parse_args()
    images = args.images or IMAGES

    rng = np.random.default_rng(args.seed)
    # min-p needs a SECOND independent null batch (Section IV-B).  It comes from a
    # separate stream so that every other number in this file is unchanged by it.
    rng_minp = np.random.default_rng([args.seed, 1001])
    n, alpha = args.n, args.alpha
    asymptotic_threshold = stats.chi2.ppf(1.0 - alpha, df=n - 1)

    print(f"n={n}, alpha={alpha}, images={', '.join(images)}")
    print(f"null draws per (image, N)={args.null_draws}, "
          f"alt draws per setting={args.alt_draws}")
    print("null p0 = image's own code histogram (non-uniform, no rank transform)")
    print()

    nulls = {}
    for name in images:
        image = load_gray_uint8(name)
        nulls[name] = code_probabilities(image, n, 0.0)
        alternatives = {eps: code_probabilities(image, n, eps)
                        for eps in args.eps if eps != 0.0}
        nulls[name] = (nulls[name], alternatives)

    header = f"{'N':>6} {'eps':>5} " + " ".join(f"{k:>7}" for k in ORDER)
    for N in args.N:
        print("=" * len(header))
        print(f"N = {N}   (mean count per bin {N / n:.1f})")
        print(header)
        null_cache = {}
        for name in images:
            p0, _ = nulls[name]
            x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
            null_cache[name] = statistics(x0, p0, N)
            x00 = rng_minp.multinomial(N, p0, size=args.null_draws).astype(float)
            null_cache[name]["_minp"] = min_p_calibration(null_cache[name], statistics(x00, p0, N), alpha)

        for eps in args.eps:
            pooled = {key: [] for key in ORDER}
            for name in images:
                p0, alternatives = nulls[name]
                p1 = p0 if eps == 0.0 else alternatives[eps]
                x1 = rng.multinomial(N, p1, size=args.alt_draws).astype(float)
                s1 = statistics(x1, p0, N)
                rates = rejection_rates({k: v for k, v in null_cache[name].items() if k != "_minp"}, s1, alpha)
                pmin, cut = null_cache[name]["_minp"]
                rates["minp"] = float(np.mean(pmin(s1) <= cut))
                rates["X2asy"] = float(np.mean(s1["X2"] > asymptotic_threshold))
                for key in ORDER:
                    pooled[key].append(rates[key])
            tag = "  (size)" if eps == 0.0 else ""
            row = " ".join(f"{np.mean(pooled[key]):7.3f}" for key in ORDER)
            print(f"{N:>6} {eps:>5.2f} {row}{tag}")
        print()


if __name__ == "__main__":
    main()
