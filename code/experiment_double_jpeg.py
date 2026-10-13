"""Double-quantization detection in DCT coefficient histograms.

The response to the reviewers of the parent paper promised that a non-uniform
null would "open the coefficient-histogram problems of image forensics and
steganalysis, where the baseline is not flat".  This is that problem.

A JPEG-like pipeline quantizes each DCT coefficient with a step q.  If an image
is compressed twice, with steps q1 then q2, the number of first-quantization
bins folded into each second-quantization bin is alternately floor(q2/q1) and
ceil(q2/q1).  For q2/q1 = 3/2 that alternation has period exactly two: the
double-quantized histogram is the single-quantized histogram multiplied by a
period-2 comb.  This is the classical double-JPEG artefact, and it is literally
the alternative the Comb Test was built for.

What makes it a problem for the comb test under a non-uniform null, rather than
for the parent uniform-null test, is the baseline.  The
histogram of a quantized AC coefficient is sharply peaked at zero and decays
roughly geometrically, so uniformity is not remotely a plausible null; the
parent paper could not have run this experiment.  Here the null is the
single-quantized histogram of the very same image, which is known exactly.

Null         single quantization with step q2.
Alternative  a fraction w of the blocks carry the extra q1 quantization, the
             rest do not.  w = 1 is a wholly recompressed image; w < 1 is a
             splice, where only a pasted region has the second history.  The
             mixture weight is the natural effect-size knob and the realistic
             forensic setting, since a forger recompresses only part of an image.
Both are computed from the same real image, so the only difference is the
processing history, which is exactly the forensic question.
"""

import argparse

import numpy as np
import skimage.color as skcolor
import skimage.data as skdata
from scipy.fft import dctn

IMAGES = ["camera", "coins", "moon", "page", "cell", "gravel", "brick"]
KEYS = ["DTV", "CT", "T(1)", "X2", "G"]


def load_gray(name):
    image = np.asarray(getattr(skdata, name)())
    if image.ndim == 3:
        image = skcolor.rgb2gray(image) * 255.0
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.float64)


def block_dct_coefficients(image, mode):
    """All 8x8 block DCT coefficients at one frequency index."""
    rows, cols = (image.shape[0] // 8) * 8, (image.shape[1] // 8) * 8
    blocks = image[:rows, :cols].reshape(rows // 8, 8, cols // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return dctn(blocks, axes=(1, 2), norm="ortho")[:, mode[0], mode[1]]


def levels(coefficients, q1, q2):
    """Quantized levels after single (q1 = 0) or double quantization."""
    if q1 > 0:
        coefficients = np.round(coefficients / q1) * q1
    return np.round(coefficients / q2).astype(np.int64)


def histogram_probabilities(values, window):
    counts = np.bincount(values + window, minlength=2 * window + 1)
    counts = counts[:2 * window + 1].astype(float)
    return counts / counts.sum()


def statistics(x, p0, N):
    expected = N * p0
    centered = x - expected
    z = centered / np.sqrt(N * p0 * (1.0 - p0))
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / expected), 0.0)
    return {
        "DTV": np.abs(np.diff(x, axis=1)).sum(1),
        "CT": np.abs(np.diff(centered, axis=1)).sum(1),
        "T(1)": np.abs(np.diff(z, axis=1)).sum(1),
        "X2": (centered ** 2 / expected).sum(1),
        "G": 2.0 * terms.sum(1),
    }


def rates(null_stats, alt_stats, alpha):
    return {key: float(np.mean(alt_stats[key] > np.quantile(null_stats[key],
                                                            1.0 - alpha)))
            for key in null_stats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, nargs=2, default=[0, 1],
                        help="DCT frequency index to test")
    parser.add_argument("--window", type=int, default=6,
                        help="tested levels are -window .. +window")
    parser.add_argument("--q1", type=int, nargs="+", default=[2, 3, 4, 6],
                        help="first quantization steps; q2 = 3*q1/2")
    parser.add_argument("--N", type=int, nargs="+", default=[500, 2000, 10000])
    parser.add_argument("--mix", type=float, nargs="+",
                        default=[0.0, 0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=40000)
    parser.add_argument("--alt_draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    window, mode = args.window, tuple(args.mode)
    print(f"DCT mode {mode}, levels -{window}..{window} ({2 * window + 1} bins), "
          f"alpha={args.alpha}")
    print("null = single quantization with q2; "
          "alternative = q1 then q2, with q2/q1 = 3/2\n")

    coefficients = {name: block_dct_coefficients(load_gray(name), mode)
                    for name in IMAGES}

    for q1 in args.q1:
        q2 = 3 * q1 // 2
        if 2 * q2 != 3 * q1:
            continue
        usable = []
        for name in IMAGES:
            c = coefficients[name]
            p0 = histogram_probabilities(np.clip(levels(c, 0, q2), -window, window),
                                         window)
            p1 = histogram_probabilities(np.clip(levels(c, q1, q2), -window, window),
                                         window)
            if p0.min() > 0 and p1.min() > 0:
                usable.append((name, p0, p1))
        if not usable:
            print(f"q1={q1}, q2={q2}: no image has all bins occupied\n")
            continue
        print(f"q1={q1} -> q2={q2}   images={len(usable)}")
        print(f"{'N':>7} {'w':>6} {'l1':>7} " + " ".join(f"{k:>7}" for k in KEYS))
        for N in args.N:
            nulls = {}
            for name, p0, _ in usable:
                x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
                nulls[name] = statistics(x0, p0, N)
            for w in args.mix:
                pooled = {k: [] for k in KEYS}
                distances = []
                for name, p0, p1 in usable:
                    mixed = (1.0 - w) * p0 + w * p1
                    distances.append(np.abs(mixed - p0).sum())
                    x1 = rng.multinomial(N, mixed,
                                         size=args.alt_draws).astype(float)
                    r = rates(nulls[name], statistics(x1, p0, N), args.alpha)
                    for k in KEYS:
                        pooled[k].append(r[k])
                row = " ".join(f"{np.mean(pooled[k]):7.3f}" for k in KEYS)
                tag = "  (size)" if w == 0.0 else ""
                print(f"{N:>7} {w:>6.2f} {np.mean(distances):>7.4f} {row}{tag}")
            print()
        print()


if __name__ == "__main__":
    main()
