"""Three follow-up experiments on the real-image non-uniform null.

A  defect geometry.  The width-mismatch defect of experiment_real_images.py is
   multiplicative: a widened code captures mass in proportion to the local signal
   density, so the comb amplitude follows the baseline envelope.  A leakage-type
   defect is additive: a fixed amount of mass is displaced between neighbouring
   codes irrespective of density.  Standardizing by sqrt(N p_i (1-p_i)) should
   help in the additive case and hurt in the multiplicative one.  Both are
   matched in l1 distance from the null so the comparison is fair.

B  sparsity.  R1's objection to the parent paper was that DTV "does not divide by
   the cell frequency" and therefore degrades when n is large relative to N.  The
   mean count per bin is swept from about 16 down to about 1 to locate the
   crossover between the CT (d_r) and T(1).

C  null misspecification.  Everything so far assumes p0 is known exactly, which
   it is when the source data are available.  In practice p0 may be estimated
   from a finite calibration capture of M0 samples.  Size and power are measured
   as a function of M0 to show how much calibration data the test needs.

All critical values are Monte-Carlo calibrated, so every entry is a rejection
rate at the same actual size.
"""

import argparse

import numpy as np
import skimage.color as skcolor
import skimage.data as skdata

FULL_AT_64 = ["camera", "coins", "moon", "page", "cell"]
FULL_AT_128 = ["camera", "page", "cell"]


def load_gray_uint8(name):
    image = np.asarray(getattr(skdata, name)())
    if image.ndim == 3:
        image = skcolor.rgb2gray(image) * 255.0
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def quantise(v, n, eps):
    widths = 1.0 + eps * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    edges = 256.0 * np.concatenate(([0.0], np.cumsum(widths / widths.sum())))
    edges[-1] = 256.0
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n - 1)


def code_probabilities(image, n, eps, rng):
    v = image.ravel().astype(np.float64) + rng.random(image.size)
    counts = np.bincount(quantise(v, n, eps), minlength=n).astype(float)
    return counts / counts.sum()


def additive_comb(p0, l1_distance):
    """p0 + (-1)^i delta, renormalized, scaled to a given l1 distance."""
    signs = np.where(np.arange(len(p0)) % 2 == 0, 1.0, -1.0)
    delta = signs - signs.sum() * p0
    delta = delta * l1_distance / np.abs(delta).sum()
    scale = 1.0
    while (p0 + scale * delta).min() <= 0.0:
        scale *= 0.5
    return p0 + scale * delta, scale


def statistics(x, p0, N):
    expected = N * p0
    centered = x - expected
    z = centered / np.sqrt(N * p0 * (1.0 - p0))
    return {
        "DTV": np.abs(np.diff(x, axis=1)).sum(1),
        "CT": np.abs(np.diff(centered, axis=1)).sum(1),
        "T(1)": np.abs(np.diff(z, axis=1)).sum(1),
        "X2": (centered ** 2 / expected).sum(1),
    }


def rates(null_stats, alt_stats, alpha):
    return {key: float(np.mean(alt_stats[key] > np.quantile(null_stats[key],
                                                            1.0 - alpha)))
            for key in null_stats}


KEYS = ["DTV", "CT", "T(1)", "X2"]


def part_a(rng, args):
    print("=" * 78)
    print("A  multiplicative (code-width mismatch) vs additive (leakage) comb,")
    print("   matched in l1 distance, n=64, pooled over 5 images")
    print("=" * 78)
    print(f"{'N':>6} {'eps':>5} {'geometry':>16} " + " ".join(f"{k:>7}" for k in KEYS)
          + f" {'l1':>7}")
    images = {name: load_gray_uint8(name) for name in FULL_AT_64}
    n = 64
    for N in args.N:
        for eps in args.eps:
            pooled = {"multiplicative": {k: [] for k in KEYS},
                      "additive": {k: [] for k in KEYS}}
            distances = []
            for name in FULL_AT_64:
                p0 = code_probabilities(images[name], n, 0.0, rng)
                p_mul = code_probabilities(images[name], n, eps, rng)
                l1 = np.abs(p_mul - p0).sum()
                p_add, _ = additive_comb(p0, l1)
                distances.append(l1)
                x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
                s0 = statistics(x0, p0, N)
                for label, p1 in (("multiplicative", p_mul), ("additive", p_add)):
                    x1 = rng.multinomial(N, p1, size=args.alt_draws).astype(float)
                    r = rates(s0, statistics(x1, p0, N), args.alpha)
                    for k in KEYS:
                        pooled[label][k].append(r[k])
            for label in ("multiplicative", "additive"):
                row = " ".join(f"{np.mean(pooled[label][k]):7.3f}" for k in KEYS)
                print(f"{N:>6} {eps:>5.2f} {label:>16} {row} "
                      f"{np.mean(distances):7.4f}")
        print()


def part_b(rng, args):
    print("=" * 78)
    print("B  sparsity sweep: mean count per bin from ~16 down to ~1")
    print("   multiplicative comb, eps=0.10, pooled over camera/page/cell")
    print("=" * 78)
    print(f"{'n':>5} {'N':>6} {'N/n':>6} " + " ".join(f"{k:>7}" for k in KEYS))
    images = {name: load_gray_uint8(name) for name in FULL_AT_128}
    eps = 0.10
    for n in (32, 64, 128):
        for N in (n // 2, n, 2 * n, 4 * n, 16 * n):
            pooled = {k: [] for k in KEYS}
            for name in FULL_AT_128:
                p0 = code_probabilities(images[name], n, 0.0, rng)
                p1 = code_probabilities(images[name], n, eps, rng)
                x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
                x1 = rng.multinomial(N, p1, size=args.alt_draws).astype(float)
                r = rates(statistics(x0, p0, N), statistics(x1, p0, N), args.alpha)
                for k in KEYS:
                    pooled[k].append(r[k])
            row = " ".join(f"{np.mean(pooled[k]):7.3f}" for k in KEYS)
            print(f"{n:>5} {N:>6} {N / n:>6.1f} {row}")
        print()


def part_c(rng, args):
    print("=" * 78)
    print("C  null estimated from a finite calibration capture of M0 samples")
    print("   n=64, N=2000, eps=0 rows are size, eps=0.10 rows are power")
    print("=" * 78)
    print(f"{'M0':>9} {'M0/N':>7} {'eps':>5} " + " ".join(f"{k:>7}" for k in KEYS))
    images = {name: load_gray_uint8(name) for name in FULL_AT_64}
    n, N, eps = 64, 2000, 0.10
    for m0 in args.M0:
        for use_eps in (0.0, eps):
            pooled = {k: [] for k in KEYS}
            for name in FULL_AT_64:
                p0 = code_probabilities(images[name], n, 0.0, rng)
                p1 = p0 if use_eps == 0.0 else code_probabilities(images[name], n,
                                                                 use_eps, rng)
                # calibration capture: M0 draws from the true ideal quantiser
                counts = rng.multinomial(m0, p0)
                if counts.min() == 0:          # unoccupied code in calibration
                    counts = counts + 1        # add-one, the standard fix
                p_hat = counts / counts.sum()
                x0 = rng.multinomial(N, p_hat, size=args.null_draws).astype(float)
                x1 = rng.multinomial(N, p1, size=args.alt_draws).astype(float)
                r = rates(statistics(x0, p_hat, N), statistics(x1, p_hat, N),
                          args.alpha)
                for k in KEYS:
                    pooled[k].append(r[k])
            row = " ".join(f"{np.mean(pooled[k]):7.3f}" for k in KEYS)
            print(f"{m0:>9} {m0 / N:>7.1f} {use_eps:>5.2f} {row}")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", default="abc")
    parser.add_argument("--N", type=int, nargs="+", default=[500, 2000, 10000])
    parser.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.10])
    parser.add_argument("--M0", type=int, nargs="+",
                        default=[2000, 10000, 50000, 250000])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=40000)
    parser.add_argument("--alt_draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=30)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if "a" in args.parts:
        part_a(rng, args)
    if "b" in args.parts:
        part_b(rng, args)
    if "c" in args.parts:
        part_c(rng, args)


if __name__ == "__main__":
    main()
