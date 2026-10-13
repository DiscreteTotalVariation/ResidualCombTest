"""When does the uncentered DTV d_u actually degenerate?  A computable a-priori
diagnostic.  (DTV below means the uncentered d_u; CT means the comb test on the
centered DTV d_r.)

The degeneracy of the uncentered d_u is not an all-or-nothing property of "non-uniform"
nulls.  It is governed by how far each adjacent pair of cells is separated
relative to the sampling noise on that pair,

    rho_i = N |p_i - p_{i-1}| / sqrt( N (p_i + p_{i-1} - (p_i - p_{i-1})^2) ) ,

which is computable from the null and N alone, before any data are seen.  Pairs
with large rho_i have a locked sign and contribute a LINEAR term to the DTV;
pairs with small rho_i still contribute a folded absolute value and keep their
comb sensitivity.

This matters because the two regimes look completely different in practice:
  - a steep, consistently sloped null (geometric run lengths, a ramp, a comb)
    locks nearly every pair, and d_u stops being a comb test entirely;
  - a natural image code histogram is bumpy, with many nearly equal adjacent
    cells, so a large fraction of pairs never lock and d_u degrades
    gently rather than catastrophically.

Reporting only one of these would be misleading, so this script reports the
diagnostic and the measured phase asymmetry side by side for every null used
anywhere in the paper.  `locked_frac` is the fraction of adjacent pairs with
rho_i > 2, i.e. a two-sigma separation.
"""

import argparse

import numpy as np
import skimage.color as skcolor
import skimage.data as skdata

IMAGES = ["camera", "coins", "moon", "page", "cell"]


def load_gray_uint8(name):
    image = np.asarray(getattr(skdata, name)())
    if image.ndim == 3:
        image = skcolor.rgb2gray(image) * 255.0
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def image_null(name, n, rng):
    image = load_gray_uint8(name)
    v = image.ravel().astype(np.float64) + rng.random(image.size)
    codes = np.clip((v * n / 256.0).astype(int), 0, n - 1)
    counts = np.bincount(codes, minlength=n).astype(float)
    return counts / counts.sum()


def rho_profile(p0, N):
    d = np.diff(p0)
    variance = N * (p0[1:] + p0[:-1] - d ** 2)
    return N * np.abs(d) / np.sqrt(variance)


def comb(p0, phase, distance):
    signs = phase * (-1.0) ** np.arange(len(p0))
    delta = p0 * signs
    delta = delta - delta.sum() * p0
    delta = delta * distance / np.abs(delta).sum()
    if (p0 + delta).min() <= 0.0:
        headroom = np.min(p0[delta < 0] / -delta[delta < 0])
        delta = delta * headroom * 0.999
    return p0 + delta


def statistics(x, p0, N):
    centered = x - N * p0
    return {"DTV": np.abs(np.diff(x, axis=1)).sum(1),
            "CT": np.abs(np.diff(centered, axis=1)).sum(1)}


def linearity(p0, N, draws, rng):
    """corr(DTV, its sign-locked linear surrogate) under the null.

    This is the diagnostic that actually works.  If every sign(x_i - x_{i-1})
    were locked to sign(p_i - p_{i-1}), the DTV would equal
    sum_i sign(p_i - p_{i-1})(x_i - x_{i-1}) identically.  The correlation
    between the two therefore measures how far the DTV has collapsed into a
    linear functional -- and a linear functional cannot be phase-blind.

    It needs no alternative, only the null and N, so a practitioner can compute
    it before collecting data and know whether the uncentered d_u is safe.
    """
    x = rng.multinomial(N, p0, size=draws).astype(float)
    differences = np.diff(x, axis=1)
    surrogate = (differences * np.sign(np.diff(p0))).sum(1)
    dtv = np.abs(differences).sum(1)
    if surrogate.std() == 0 or dtv.std() == 0:
        # A flat null has no locked direction at all, so the linear part is
        # exactly zero rather than undefined.
        return 0.0
    return float(np.corrcoef(dtv, surrogate)[0, 1])


def phase_ratio(p0, N, distance, alpha, draws, rng):
    x0 = rng.multinomial(N, p0, size=draws).astype(float)
    s0 = statistics(x0, p0, N)
    cuts = {k: np.quantile(s0[k], 1.0 - alpha) for k in s0}
    power = {}
    for phase in (1.0, -1.0):
        x1 = rng.multinomial(N, comb(p0, phase, distance),
                             size=draws).astype(float)
        s1 = statistics(x1, p0, N)
        power[phase] = {k: float(np.mean(s1[k] > cuts[k])) for k in s1}
    out = {}
    for key in ("DTV", "CT"):
        values = [power[1.0][key], power[-1.0][key]]
        out[key] = (max(values) / max(min(values), 1e-9), min(values))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=[500, 2000, 10000])
    parser.add_argument("--distance", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=40000)
    parser.add_argument("--n_image", type=int, default=64)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    nulls = {
        "uniform n=8": np.ones(8) / 8,
        "tent 1:2:3:4:3:2:1": np.array([1., 2, 3, 4, 3, 2, 1]) / 16,
        "ramp 1:2:3:4:5:6": np.array([1., 2, 3, 4, 5, 6]) / 21,
        "comb 1:2:1:2:1": np.array([1., 2, 1, 2, 1]) / 7,
        "run-length 16:8:4:2:1:1": np.array([16., 8, 4, 2, 1, 1]) / 32,
    }
    for name in IMAGES:
        nulls[f"image {name} n={args.n_image}"] = image_null(
            name, args.n_image, rng)

    print(f"comb at l1={args.distance}, alpha={args.alpha}, "
          f"{args.draws} draws per cell")
    print("rho_min/rho_med over adjacent pairs; locked_frac = fraction with "
          "rho_i > 2")
    print("ratio = max/min power over the two comb phases; worst = the "
          "smaller of the two\n")

    for N in args.N:
        print(f"N = {N}")
        print(f"{'null':>26} {'rho_med':>8} {'locked':>7} {'linearity':>10} | "
              f"{'DTV ratio':>12} {'DTV worst':>10} | {'CT worst':>11}")
        rows = []
        for name, p0 in nulls.items():
            profile = rho_profile(p0, N)
            locked = float(np.mean(profile > 2.0))
            linear = linearity(p0, N, args.draws, rng)
            measured = phase_ratio(p0, N, args.distance, args.alpha,
                                   args.draws, rng)
            rows.append((linear, measured["DTV"][0], measured["DTV"][1]))
            print(f"{name:>26} {np.median(profile):>8.2f} {locked:>7.2f} "
                  f"{linear:>10.3f} | {measured['DTV'][0]:>12.1f} "
                  f"{measured['DTV'][1]:>10.3f} | "
                  f"{measured['CT'][1]:>11.3f}")
        # The index ORDERS the loss of phase-blindness; it is not a cut-off.  An
        # earlier version of this script reported a split at 0.75, which is an
        # artifact of these ten nulls: their indices cluster below 0.51 and above
        # 0.89, and experiment_rho_sweep.py fills the gap and finds a monotone
        # decline with no clean cut below about 0.91.
        index_values = np.array([r[0] for r in rows])
        worst_values = np.array([r[2] for r in rows])
        rank = lambda v: np.argsort(np.argsort(v))
        spearman = float(np.corrcoef(rank(index_values), rank(-worst_values))[0, 1])
        # Note: this is rank(index) vs rank(MINUS power), hence positive; the
        # Letter and experiment_rho_sweep.py quote the negative index-vs-power sign.
        print(f"  rank correlation between the index and MINUS d_u's worse-sign "
              f"power: {spearman:.3f} (higher index, less power; see "
              f"experiment_rho_sweep.py for 42 nulls spanning the range)")
        print()


if __name__ == "__main__":
    main()
