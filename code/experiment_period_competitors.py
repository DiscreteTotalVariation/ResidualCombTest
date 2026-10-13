"""Period-2 and period-4 combs: CT against AC, X^2 and their min-p combination.

Reviewers' point: the alternating contrast AC = |sum_i (-1)^i e_i| (e = x - N p)
takes an absolute value, so it is blind to WHICH bins are raised and needs only the
period, 2.  It is therefore not an oracle in the strict sense, and a Tippett min-p
combination of AC and Pearson X^2 -- calibrated on an independent second null
batch, exactly as min_p_calibration in experiment_real_images.py -- may beat the
CT (d_r = sum |e_{i+1} - e_i|) on period-2 combs while the CT keeps an edge at
period 4.  This script measures that, in two blocks.

Block 1 (images).  The image experiment of experiment_real_images.py, per image,
with the Letter's settings: camera, coins, moon, page, cell; n = 64, N = 2000,
code widths proportional to 1 + c_i eps, eps = +-0.06, 4e4 null and 2e4
alternative draws, alpha = 0.05, MC-calibrated.  Two comb patterns:
  period 2:  c = (+1, -1, +1, -1, ...)    (the Letter's; reproduces code_edges)
  period 4:  c = (+1, +1, -1, -1, ...)
Both signs of eps; the entry is the WORSE sign, as in the Letter's tables.
Statistics: CT (= d_r), AC, X2 (unpooled), X2p (pooled to expected count 5),
minp(AC,X2) (new), and, for reference, the Letter's minp over T(0), T(0.5), T(1).

Block 2 (synthetic).  experiment_design.py --part acperiod, unchanged: four nulls,
N = 2000, l1 distance 0.10, multiplicative comb p0 * cos(2 pi i / period)
re-centred, periods 2, 3, 4, 6, 4e4 null and 2e4 alternative draws per sign,
plus minp(AC,X2) and the size of every rule.

Reproduction.  The default seeds are those of the two parent scripts (19 for the
images, 16 for acperiod), and the parent draws are taken from the SAME stream in
the SAME order, so the period-2 image rows and every acperiod row are draw-for-draw
the existing results; the script parses ../results/experiment_real_images_per_image.txt
and ../results/experiment_design_acperiod.txt and prints the differences.  Every
NEW draw (period-4 combs, min-p second batches for the synthetic block, size
batches) comes from its own child stream np.random.default_rng([seed, k]), so it
neither disturbs nor is disturbed by the parent stream.  Run with other seeds
(--seed_images, --seed_synthetic) the reproduction check becomes a check within
MC error instead of an identity.

Standard errors: the binomial sqrt(p (1 - p) / m) over the m alternative draws of
the reported sign, combined in quadrature with the MC error of the estimated
critical value (local slope d power / d size times sqrt(alpha (1 - alpha) / M),
M null draws).  For min-p only the second batch's share of that error is
included.  The worse-of-two-signs minimum is biased slightly downward when the two
signs are within MC error of each other; this is not corrected (nor is it in the
parent scripts).
"""

import argparse
import os
import platform
import sys

import numpy as np
import scipy
import skimage

import experiment_real_images as img

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SYN_NULLS = (("tent 1:2:3:4:3:2:1", (1, 2, 3, 4, 3, 2, 1)),
             ("run-length 16:8:4:2:1:1", (16, 8, 4, 2, 1, 1)),
             ("ramp 1:2:3:4:5:6", (1, 2, 3, 4, 5, 6)),
             ("geometric n=12", tuple(int(round(100 * np.exp(-0.2 * i))) for i in range(12))))


# ----------------------------------------------------------------- shared tools

def min_p_calibration(null1, null2, members, alpha):
    """Tippett's min-p over `members`, the same rule as
    experiment_real_images.min_p_calibration but with the members as an
    argument: null1 gives each member's empirical p-value map, the independent
    null2 gives the null law of the minimum, whose lower alpha-quantile is the
    cut; reject when the minimum is AT OR BELOW the cut."""
    sorted1 = {m: np.sort(null1[m]) for m in members}
    K = float(len(sorted1[members[0]]))

    def pmin(stats_dict):
        return np.min([1.0 - np.searchsorted(sorted1[m], stats_dict[m], "left") / K
                       for m in members], axis=0)

    null2_pmin = pmin(null2)
    return pmin, float(np.quantile(null2_pmin, alpha, method="lower")), null2_pmin


H = 0.01   # half-width, in null tail probability, of the slope estimate below


def upper_rate(null_values, alt_values, alpha):
    return float(np.mean(alt_values > np.quantile(null_values, 1.0 - alpha)))


def upper_cut_se(null_values, alt_values, alpha):
    """MC error of the power due to the ESTIMATED critical value: the slope
    d(power)/d(size) near the cut, from cuts at sizes alpha -+ H, times the
    binomial SE of the null tail at the cut, sqrt(alpha(1-alpha)/M)."""
    lo, hi = np.quantile(null_values, [1.0 - alpha - H, 1.0 - alpha + H])
    slope = (np.mean(alt_values > lo) - np.mean(alt_values > hi)) / (2 * H)
    return float(slope * np.sqrt(alpha * (1.0 - alpha) / len(null_values)))


def minp_rate_and_cut_se(cal, s1, alpha):
    """Rejection rate of a calibrated min-p rule and the same critical-value
    error term as upper_cut_se, taken over the second null batch only (the
    p-value maps of the first batch add a further, unestimated, error)."""
    pmin, cut, null2_pmin = cal
    alt = pmin(s1)
    lo, hi = np.quantile(null2_pmin, [alpha - H, alpha + H], method="lower")
    slope = (np.mean(alt <= hi) - np.mean(alt <= lo)) / (2 * H)
    return float(np.mean(alt <= cut)), float(slope * np.sqrt(alpha * (1.0 - alpha) / len(null2_pmin)))


def total_se(rates, key, m):
    """binomial SE over the m alternative draws combined with the critical-value term"""
    return float(np.hypot(se(rates[key], m), rates.get(key + "_cse", 0.0)))


def diff_z(d, s_total):
    """d / SE of a difference between two INDEPENDENT runs, each reported to 3
    decimals: sqrt(2 s^2 + 2 (0.001^2 / 12))."""
    sd = np.sqrt(2 * s_total ** 2 + 2 * (0.001 ** 2 / 12))
    return d / sd


def se(p, m):
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / m))


def provenance(args):
    print("experiment_period_competitors.py")
    print("command: python3 " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]))
    print(f"python {platform.python_version()}, numpy {np.__version__}, "
          f"scipy {scipy.__version__}, scikit-image {skimage.__version__}")
    print(f"seeds: images {args.seed_images} (parent stream, = experiment_real_images.py), "
          f"synthetic {args.seed_synthetic} (parent stream, = experiment_design.py); "
          "new draws on child streams default_rng([seed, k])")
    print(f"alpha={args.alpha}; CT = d_r = sum|e_(i+1) - e_i|, e = x - N p; "
          "AC = |sum (-1)^i e_i|; du = uncentered DTV d_u")
    print()


# ----------------------------------------------------------------- image block

def pattern(n, period):
    i = np.arange(n)
    if period == 2:
        return np.where(i % 2 == 0, 1.0, -1.0)
    if period == 4:
        return np.where(i % 4 < 2, 1.0, -1.0)
    raise ValueError(period)


def code_probabilities_pattern(image, n, eps, c):
    """img.code_probabilities with widths 1 + c_i eps for an arbitrary pattern c
    (same edges, same exact surrogate law)."""
    widths = 1.0 + eps * c
    edges = 256.0 * np.concatenate(([0.0], np.cumsum(widths / widths.sum())))
    edges[-1] = 256.0
    mass = np.bincount(image.ravel(), minlength=256).astype(float)
    mass /= mass.sum()
    intensity = np.arange(256, dtype=float)
    low = np.maximum(intensity[:, None], edges[None, :-1])
    high = np.minimum(intensity[:, None] + 1.0, edges[None, 1:])
    return mass @ np.clip(high - low, 0.0, None)


IMG_KEYS = ["CT", "AC", "X2", "X2p", "minpACX2", "minpT"]
IMG_LABEL = {"CT": "CT", "AC": "AC", "X2": "X2", "X2p": "X2p",
             "minpACX2": "mp(AC,X2)", "minpT": "mp(T)"}


def image_rates(null1, cal_acx2, cal_t, s1, alpha):
    out = {}
    for k in ("CT", "AC", "X2", "X2p"):
        out[k] = upper_rate(null1[k], s1[k], alpha)
        out[k + "_cse"] = upper_cut_se(null1[k], s1[k], alpha)
    out["minpACX2"], out["minpACX2_cse"] = minp_rate_and_cut_se(cal_acx2, s1, alpha)
    out["minpT"], out["minpT_cse"] = minp_rate_and_cut_se(cal_t, s1, alpha)
    return out


def parse_per_image(path):
    """{image: {eps: {column: value}}} from the per-image results file."""
    table, name, cols = {}, None, None
    with open(path) as fh:
        for line in fh:
            if line.startswith("### "):
                name = line[4:].strip(); table[name] = {}
            elif line.strip().startswith("N ") and "eps" in line:
                cols = line.split()
            elif name and cols and line.strip() and line.split()[0] == "2000":
                parts = line.split()
                table[name][float(parts[1])] = dict(zip(cols[2:], map(float, parts[2:2 + len(cols) - 2])))
    return table


def block_images(args):
    n, N, alpha, eps = args.n, args.N_images, args.alpha, args.eps
    md, ma = args.null_draws, args.alt_draws
    print("=" * 100)
    print(f"BLOCK 1: image code histograms, n={n}, N={N}, widths ~ 1 + c_i*eps, eps=+-{eps}, "
          f"{md} null / {ma} alternative draws per (image, sign)")
    print("period 2: c = +,-,+,-,...  (Letter);  period 4: c = +,+,-,-,...")
    print("worse sign per statistic; size = rejection rate on an alternative batch drawn from p0")
    print("=" * 100)
    reference = parse_per_image(os.path.join(RESULTS, "experiment_real_images_per_image.txt"))
    rows, checks = [], []
    for name in img.IMAGES:
        image = img.load_gray_uint8(name)
        p0 = img.code_probabilities(image, n, 0.0)
        assert np.allclose(code_probabilities_pattern(image, n, 0.0, pattern(n, 2)), p0, atol=1e-15)
        for e in (eps, -eps):
            assert np.allclose(code_probabilities_pattern(image, n, e, pattern(n, 2)),
                               img.code_probabilities(image, n, e), rtol=0, atol=1e-15)
        # parent streams, consumed in the parent script's order (null, min-p batch,
        # then alternatives for eps = 0, +eps, -eps)
        rng = np.random.default_rng(args.seed_images)
        rng_minp = np.random.default_rng([args.seed_images, 1001])
        rng_p4 = np.random.default_rng([args.seed_images, 2001, img.IMAGES.index(name)])
        null1 = img.statistics(rng.multinomial(N, p0, size=md).astype(float), p0, N)
        null2 = img.statistics(rng_minp.multinomial(N, p0, size=md).astype(float), p0, N)
        cal_acx2 = min_p_calibration(null1, null2, ("AC", "X2"), alpha)
        cal_t = min_p_calibration(null1, null2, ("CT", "T(0.5)", "T(1)"), alpha)
        # the generic rule must equal the parent's own min-p
        parent_pmin, parent_cut = img.min_p_calibration(null1, null2, alpha)
        assert parent_cut == cal_t[1]

        def run(p1, generator):
            s1 = img.statistics(generator.multinomial(N, p1, size=ma).astype(float), p0, N)
            return image_rates(null1, cal_acx2, cal_t, s1, alpha)

        size = run(p0, rng)
        per_sign = {2: {}, 4: {}}
        per_sign[2][+1] = run(code_probabilities_pattern(image, n, eps, pattern(n, 2)), rng)
        per_sign[2][-1] = run(code_probabilities_pattern(image, n, -eps, pattern(n, 2)), rng)
        per_sign[4][+1] = run(code_probabilities_pattern(image, n, eps, pattern(n, 4)), rng_p4)
        per_sign[4][-1] = run(code_probabilities_pattern(image, n, -eps, pattern(n, 4)), rng_p4)
        for period in (2, 4):
            worse, worse_se = {}, {}
            for k in IMG_KEYS:
                w = min((+1, -1), key=lambda sg: per_sign[period][sg][k])
                worse[k] = per_sign[period][w][k]
                worse_se[k] = total_se(per_sign[period][w], k, ma)
            rows.append((name, period, size, per_sign[period], worse, worse_se))
        # reproduction against the parent file (period 2, eps = 0, +eps, -eps)
        if name in reference:
            ref = reference[name]
            for e, mine in ((0.0, size), (eps, per_sign[2][+1]), (-eps, per_sign[2][-1])):
                if e not in ref:
                    continue
                for key, col in (("CT", "CT"), ("AC", "AC"), ("X2", "X2"),
                                 ("X2p", "X2p"), ("minpT", "minp")):
                    checks.append((name, e, col, ref[e][col], mine[key], total_se(mine, key, ma)))

    print()
    print(f"{'image':>7} {'per':>3} " + " ".join(f"{IMG_LABEL[k]:>10}" for k in IMG_KEYS)
          + "   (worse-sign power)")
    for name, period, size, signs, worse, _ in rows:
        print(f"{name:>7} {period:>3} " + " ".join(f"{worse[k]:>10.3f}" for k in IMG_KEYS))
    for period in (2, 4):
        sel = [w for _, per, _, _, w, _ in rows if per == period]
        print(f"{'mean':>7} {period:>3} " + " ".join(f"{np.mean([w[k] for w in sel]):>10.3f}" for k in IMG_KEYS))
        print(f"{'min':>7} {period:>3} " + " ".join(f"{np.min([w[k] for w in sel]):>10.3f}" for k in IMG_KEYS))
    print()
    print("MC standard errors of the worse-sign entries above: binomial over the alternative")
    print("draws combined with the error of the MC critical value (see upper_cut_se):")
    for name, period, size, signs, worse, worse_se in rows:
        print(f"{name:>7} {period:>3} " + " ".join(f"{worse_se[k]:>10.4f}" for k in IMG_KEYS))
    print()
    print("both signs (+eps / -eps) for completeness:")
    for name, period, size, signs, worse, _ in rows:
        print(f"{name:>7} {period:>3} " + " ".join(
            f"{signs[+1][k]:.3f}/{signs[-1][k]:.3f}".rjust(12) for k in IMG_KEYS))
    print()
    print(f"size of each rule (eps = 0, {ma} draws, SE ~ {se(alpha, ma):.4f}):")
    for name, period, size, signs, worse, _ in rows:
        if period == 2:
            print(f"{name:>7}     " + " ".join(f"{size[k]:>10.3f}" for k in IMG_KEYS))
    print()
    print("REPRODUCTION CHECK vs results/experiment_real_images_per_image.txt "
          "(period 2, same seed and draw order => expected identical to 3 decimals)")
    print("diff/SE uses the SE of a difference of two independent runs (only meaningful with")
    print("non-default seeds; with the default seeds every diff must be exactly 0):")
    print(f"{'image':>7} {'eps':>6} {'stat':>6} {'file':>7} {'here':>7} {'diff':>7} {'diff/SE':>8}")
    worst, max_abs = 0.0, 0.0
    for name, e, col, ref_value, mine, s_total in checks:
        d = round(mine, 3) - ref_value
        z = diff_z(d, s_total)
        max_abs = max(max_abs, abs(d))
        worst = max(worst, abs(z))
        print(f"{name:>7} {e:>6.2f} {col:>6} {ref_value:>7.3f} {mine:>7.3f} {d:>+7.3f} {z:>+8.2f}")
    print(f"{len(checks)} values compared; max |diff| = {max_abs:.3f}, max |diff/SE| = {worst:.2f}")
    print()


# ------------------------------------------------------------- synthetic block

def syn_stats(x, p0, N, signs2):
    e = x - N * p0
    return {"CT": np.abs(np.diff(e, axis=1)).sum(1),
            "AC": np.abs(e @ signs2),
            "X2": (e ** 2 / (N * p0)).sum(1),
            "du": np.abs(np.diff(x, axis=1)).sum(1)}


def parse_acperiod(path):
    """Positional: the last four numbers of a row are CT, AC, X2, du."""
    table = {}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 6 and parts[-5].isdigit():
                try:
                    vals = list(map(float, parts[-4:]))
                except ValueError:
                    continue
                table[(" ".join(parts[:-5]), int(parts[-5]))] = dict(zip(("CT", "AC", "X2", "du"), vals))
    return table


SYN_KEYS = ["CT", "AC", "X2", "minpACX2", "du"]
SYN_LABEL = {"CT": "CT", "AC": "AC", "X2": "X2", "minpACX2": "mp(AC,X2)", "du": "du"}


def block_synthetic(args):
    N, alpha, draws = args.N_synthetic, args.alpha, args.draws_synthetic
    print("=" * 100)
    print(f"BLOCK 2: synthetic nulls of experiment_design.py --part acperiod, N={N}, "
          f"l1 distance {args.distance}, delta ~ p0*cos(2 pi i/period) re-centred,")
    print(f"{draws} null draws, {draws // 2} alternative draws per sign; worse sign. "
          "du is the uncentered DTV d_u (column 'du' of the parent file).")
    print("=" * 100)
    reference = parse_acperiod(os.path.join(RESULTS, "experiment_design_acperiod.txt"))
    rng = np.random.default_rng(args.seed_synthetic)               # parent stream
    rows, sizes, checks = [], [], []
    for index, (name, weights) in enumerate(SYN_NULLS):
        rng_minp = np.random.default_rng([args.seed_synthetic, 1001, index])
        rng_size = np.random.default_rng([args.seed_synthetic, 3001, index])
        q = np.array(weights, float); p0 = q / q.sum(); n = len(q)
        signs2 = (-1.0) ** np.arange(n)
        null1 = syn_stats(rng.multinomial(N, p0, size=draws).astype(float), p0, N, signs2)
        null2 = syn_stats(rng_minp.multinomial(N, p0, size=draws).astype(float), p0, N, signs2)
        cal = min_p_calibration(null1, null2, ("AC", "X2"), alpha)

        def rates(s1):
            out = {}
            for k in ("CT", "AC", "X2", "du"):
                out[k] = upper_rate(null1[k], s1[k], alpha)
                out[k + "_cse"] = upper_cut_se(null1[k], s1[k], alpha)
            out["minpACX2"], out["minpACX2_cse"] = minp_rate_and_cut_se(cal, s1, alpha)
            return out

        sizes.append((name, rates(syn_stats(rng_size.multinomial(N, p0, size=draws // 2).astype(float),
                                            p0, N, signs2))))
        for period in (2, 3, 4, 6):
            shape = np.cos(2 * np.pi * np.arange(n) / period)
            delta = p0 * shape
            delta = delta - delta.sum() * p0
            scale = np.abs(delta).sum()
            if scale == 0:
                continue
            delta = delta * args.distance / scale
            if (p0 + delta).min() <= 0:
                room = np.min(p0[delta < 0] / -delta[delta < 0])
                delta = delta * room * 0.999
            signs = {}
            for sign in (+1, -1):
                x1 = rng.multinomial(N, p0 + sign * delta, size=draws // 2).astype(float)
                signs[sign] = rates(syn_stats(x1, p0, N, signs2))
            worse, worse_se = {}, {}
            for k in SYN_KEYS:
                w = min((+1, -1), key=lambda sg: signs[sg][k])
                worse[k] = signs[w][k]
                worse_se[k] = total_se(signs[w], k, draws // 2)
            rows.append((name, period, signs, worse, worse_se))
            if (name, period) in reference:
                for k in ("CT", "AC", "X2", "du"):
                    checks.append((name, period, k, reference[(name, period)][k], worse[k], worse_se[k]))

    m = draws // 2
    print()
    print(f"{'null':>24} {'per':>3} " + " ".join(f"{SYN_LABEL[k]:>10}" for k in SYN_KEYS)
          + "   (worse-sign power)")
    for name, period, signs, worse, _ in rows:
        print(f"{name:>24} {period:>3} " + " ".join(f"{worse[k]:>10.3f}" for k in SYN_KEYS))
    print()
    print("MC standard errors (binomial over alternative draws combined with critical-value error):")
    for name, period, signs, worse, worse_se in rows:
        print(f"{name:>24} {period:>3} " + " ".join(f"{worse_se[k]:>10.4f}" for k in SYN_KEYS))
    print()
    print("both signs (+ / -):")
    for name, period, signs, worse, _ in rows:
        print(f"{name:>24} {period:>3} " + " ".join(
            f"{signs[+1][k]:.3f}/{signs[-1][k]:.3f}".rjust(12) for k in SYN_KEYS))
    print()
    print(f"size of each rule ({m} fresh null draws from a separate stream, SE ~ {se(alpha, m):.4f}):")
    for name, s in sizes:
        print(f"{name:>24}     " + " ".join(f"{s[k]:>10.3f}" for k in SYN_KEYS))
    print()
    print("REPRODUCTION CHECK vs results/experiment_design_acperiod.txt "
          "(same seed and draw order => expected identical to 3 decimals)")
    print("diff/SE as in block 1 (meaningful only with non-default seeds):")
    print(f"{'null':>24} {'per':>3} {'stat':>7} {'file':>7} {'here':>7} {'diff':>7} {'diff/SE':>8}")
    worst, max_abs = 0.0, 0.0
    for name, period, k, ref_value, mine, s_total in checks:
        d = round(mine, 3) - ref_value
        z = diff_z(d, s_total)
        worst = max(worst, abs(z)); max_abs = max(max_abs, abs(d))
        print(f"{name:>24} {period:>3} {k:>7} {ref_value:>7.3f} {mine:>7.3f} {d:>+7.3f} {z:>+8.2f}")
    print(f"{len(checks)} values compared; max |diff| = {max_abs:.3f}, max |diff/SE| = {worst:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--block", choices=("images", "synthetic", "both"), default="both")
    parser.add_argument("--seed_images", type=int, default=19,
                        help="parent seed of experiment_real_images.py (draw-for-draw reproduction)")
    parser.add_argument("--seed_synthetic", type=int, default=16,
                        help="parent seed of experiment_design.py (draw-for-draw reproduction)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--N_images", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=0.06)
    parser.add_argument("--null_draws", type=int, default=40000)
    parser.add_argument("--alt_draws", type=int, default=20000)
    parser.add_argument("--N_synthetic", type=int, default=2000)
    parser.add_argument("--distance", type=float, default=0.10)
    parser.add_argument("--draws_synthetic", type=int, default=40000)
    args = parser.parse_args()
    provenance(args)
    if args.block in ("images", "both"):
        block_images(args)
    if args.block in ("synthetic", "both"):
        block_synthetic(args)


if __name__ == "__main__":
    main()
