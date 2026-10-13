"""Large-scale stress tests of the Letter's near-universal claims.

Each claim below is stated in the Letter for a handful of nulls.  This script
re-tests each one over a large randomized family, so that a claim is either
confirmed at scale or has its exceptions catalogued.  Run one part at a time:

  --part exact    the exact null is correct        (DP vs independent brute force)
  --part phase    the CT is phase-blind and the uncentered DTV is not
                  (contribution ii)
  --part bias     neither test is unbiased         (magnitudes, at scale)
  --part gamma    the gamma false positive rate    (many nulls, large N)
  --part images   the CT beats chi-square on real images (all usable images)
  --part altpath  the ALTERNATIVE-centred DP path, which produces every exact
                  power in Table I, against independent brute force
  --part edge     degenerate inputs: n = 1, 2, N = 0, 1, repeated and extreme
                  weights, interior zero slopes, N a multiple of L and not
  --part levels   the gamma approximation away from alpha = 0.05, down to 0.001,
                  which is where a referee recalibrating the test would look
  --part misspec  the Conclusions' null-misspecification sizes, averaged over
                  many calibration draws instead of one
  --part indexn   the collapse index and its ordering across N, not only N = 2000
  --part cbound   the bound c <= min(L, 2^(n-1)) over random weights
  --part printed  the recurrence AS PRINTED in the Letter (equation (4)),
                  transcribed literally and evaluated, against the code and
                  against brute force: catches a typo in the typeset formula
  --part pvalue   is the exact p-value a VALID p-value: super-uniform under the
                  null, so that the level is never exceeded (an off-by-one in the
                  tail would show here and nowhere else)
  --part local    WHY the CT can be biased: the first-order change of
                  E[d_r] in the comb direction, which the variance can make
                  negative while the mean shift only helps at second order

Everything is exact where exactness is available and Monte Carlo otherwise, and
every part prints the WORST case it found, not an average.
"""

import argparse
import itertools
from fractions import Fraction as F
from math import factorial

import numpy as np

from du_exact import (calculate_exact_dtv_distribution_nonuniform as du_law,
                   prepare_integer_weights)
from dr_exact import calculate_exact_dr_distribution as dr_law


# ------------------------------------------------------------------ helpers --

def tail(distribution, total, alpha):
    survival = total
    for value, count in distribution:
        survival -= count
        if F(survival, total) <= alpha:
            return value, F(survival, total)
    return distribution[-1][0], F(0)


def exceedance(distribution, total, threshold):
    return F(sum(c for v, c in distribution if v > threshold), total)


def brute_force(N, q, q_centre=None):
    """Independent enumeration: every histogram once, with its multinomial weight."""
    q_centre = q_centre or q
    n, L = len(q), sum(q_centre)
    totals = {}
    for cut in itertools.combinations(range(N + n - 1), n - 1):
        histogram, previous = [], -1
        for c in cut:
            histogram.append(c - previous - 1)
            previous = c
        histogram.append(N + n - 1 - previous - 1)
        weight = factorial(N)
        for count, wi in zip(histogram, q):
            weight //= factorial(count)
            weight *= wi ** count
        value = sum(abs(L * (histogram[i] - histogram[i - 1])
                        - N * (q_centre[i] - q_centre[i - 1])) for i in range(1, n))
        totals[value] = totals.get(value, 0) + weight
    return totals


def random_rational_null(rng, n_range=(2, 7), weight_range=(1, 12)):
    n = int(rng.integers(*n_range))
    return tuple(int(v) for v in rng.integers(weight_range[0], weight_range[1], size=n))


def comb_pair(p0, distance):
    """The two signs of a period-2 comb at l1 `distance`, as probability vectors."""
    signs = (-1.0) ** np.arange(len(p0))
    delta = p0 * signs
    delta = delta - delta.sum() * p0
    delta = delta * distance / np.abs(delta).sum()
    if (p0 + delta).min() <= 0 or (p0 - delta).min() <= 0:
        room = np.min(p0[np.abs(delta) > 0] / np.abs(delta)[np.abs(delta) > 0])
        delta = delta * room * 0.999
    return p0 + delta, p0 - delta


# -------------------------------------------------------------------- parts --

def part_exact(args):
    """Claim: the DP gives the exact null for ANY rational non-uniform null."""
    rng = np.random.default_rng(args.seed)
    print("DP vs independent brute force on random rational nulls; also total mass = L^N")
    print(f"{'case':>5} {'q':>26} {'N':>4} {'L':>5} {'support':>8} {'mass ok':>8} {'agrees':>7}")
    bad = 0
    for case in range(args.cases):
        q = random_rational_null(rng)
        n = len(q)
        N = int(rng.integers(2, 11 if n > 4 else 14))
        law = dr_law(N, list(q))
        fast = {v: c for v, c in law if c}
        slow = brute_force(N, list(q))
        mass = sum(fast.values()) == sum(q) ** N
        agree = fast == slow
        bad += not (mass and agree)
        if not (mass and agree) or case < 6 or case % 25 == 0:
            print(f"{case:>5} {str(q):>26} {N:>4} {sum(q):>5} {len(fast):>8} "
                  f"{'yes' if mass else 'NO':>8} {'yes' if agree else 'NO':>7}")
    print(f"\n{args.cases} random rational nulls: {bad} disagreements with brute force")


def part_phase(args):
    """Claim (ii): the CT (statistic d_r) is phase-blind; the uncentered d_u is not.

    Phase-blindness is measured as the ratio of the two comb signs' powers, which
    a phase-blind test keeps at 1.  Exact for small N, Monte Carlo otherwise.
    """
    rng = np.random.default_rng(args.seed)
    print("ratio = max/min power over the two comb signs (1.0 = phase-blind); "
          f"{args.draws} draws, alpha={args.alpha}")
    print(f"{'q':>24} {'N':>5} {'l1':>6} {'CT ratio':>10} {'du ratio':>9} {'CT worst':>10} {'du worst':>9}")
    worst_ct_ratio, worst_case = 1.0, None
    rows = 0
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(3, 9), weight_range=(1, 20))
        if len(set(q)) == 1:
            continue
        p0 = np.array(q, float) / sum(q)
        N = int(rng.choice([200, 500, 1000, 2000]))
        distance = float(rng.choice([0.04, 0.08, 0.12, 0.20]))
        plus, minus = comb_pair(p0, distance)
        if min(plus.min(), minus.min()) <= 0:
            continue
        x0 = rng.multinomial(N, p0, size=args.draws)
        r0 = x0 - N * p0
        cut_w = np.quantile(np.abs(np.diff(x0, axis=1)).sum(1), 1 - args.alpha)
        cut_r = np.quantile(np.abs(np.diff(r0, axis=1)).sum(1), 1 - args.alpha)
        powers = {}
        for name, p1 in (("+", plus), ("-", minus)):
            x1 = rng.multinomial(N, p1, size=args.draws)
            r1 = x1 - N * p0
            powers[name] = (float(np.mean(np.abs(np.diff(x1, axis=1)).sum(1) > cut_w)),
                            float(np.mean(np.abs(np.diff(r1, axis=1)).sum(1) > cut_r)))
        w_hi, w_lo = max(powers["+"][0], powers["-"][0]), min(powers["+"][0], powers["-"][0])
        r_hi, r_lo = max(powers["+"][1], powers["-"][1]), min(powers["+"][1], powers["-"][1])
        ratio_w = w_hi / max(w_lo, 1e-9)
        ratio_r = r_hi / max(r_lo, 1e-9)
        rows += 1
        if ratio_r > worst_ct_ratio:
            worst_ct_ratio, worst_case = ratio_r, (q, N, distance, r_hi, r_lo)
        if rows <= 12 or ratio_r > 1.5:
            print(f"{str(q):>24} {N:>5} {distance:>6.2f} {ratio_r:>10.2f} {ratio_w:>9.1f} "
                  f"{r_lo:>10.3f} {w_lo:>9.3f}")
    print(f"\n{rows} (null, N, distance) cases.  Worst CT phase ratio {worst_ct_ratio:.2f}"
          + (f" at q={worst_case[0]}, N={worst_case[1]}, l1={worst_case[2]:.2f} "
             f"(powers {worst_case[3]:.3f} and {worst_case[4]:.3f})" if worst_case else ""))


def part_bias(args):
    """Claim: neither test is unbiased, but the CT's shortfall is negligible.

    Exact size and power against period-2 combs over many random rational nulls
    and a fine grid of amplitudes; the SHORTFALL (size minus power) is what
    matters, not the count.
    """
    rng = np.random.default_rng(args.seed)
    alpha = F(args.alpha).limit_denominator(10 ** 6)
    grid = [(100, 1), (100, 2), (50, 2), (100, 5), (20, 1), (100, 10), (10, 2),
            (5, 1), (4, 1), (3, 1), (2, 1)]
    print("exact size and power against period-2 combs, both signs; shortfall = size - power")
    print(f"{'q':>22} {'N':>4} {'combs':>6} {'CT n':>6} {'CT max':>8} {'at eps':>7} "
          f"{'du n':>5} {'du max':>8} {'at eps':>7}")
    worst_r = (0.0, None); worst_r_eps = 0.0
    for case in range(args.cases):
        q0 = random_rational_null(rng, n_range=(3, 7), weight_range=(1, 9))
        if len(set(q0)) == 1:
            continue
        L0, n = sum(q0), len(q0)
        N = int(rng.choice([12, 16, 20, 25, 30, 40]))
        crit_r, size_r = tail(dr_law(N, list(q0)), L0 ** N, alpha)
        crit_w, size_w = tail(du_law(N, list(q0)), L0 ** N, alpha)
        nr = nw = total = 0
        max_r = max_w = 0.0; eps_r = eps_w = 0.0
        for den, num in grid:
            for sign in (+1, -1):
                q1 = tuple(v * (den + (sign if i % 2 == 0 else -sign) * num)
                           for i, v in enumerate(q0))
                if min(q1) <= 0:
                    continue
                total += 1
                eps = num / den
                pr = exceedance(dr_law(N, list(q1), list(q0)), sum(q1) ** N, crit_r)
                pw = exceedance(du_law(N, list(q1)), sum(q1) ** N, crit_w)
                if pr < size_r:
                    nr += 1
                    if float(size_r - pr) > max_r:
                        max_r, eps_r = float(size_r - pr), eps
                if pw < size_w:
                    nw += 1
                    if float(size_w - pw) > max_w:
                        max_w, eps_w = float(size_w - pw), eps
        if max_r > worst_r[0]:
            worst_r = (max_r, (q0, N, eps_r, float(size_r)))
        worst_r_eps = max(worst_r_eps, eps_r)
        print(f"{str(q0):>22} {N:>4} {total:>6} {nr:>6} {max_r:>8.4f} {eps_r:>7.2f} "
              f"{nw:>5} {max_w:>8.4f} {eps_w:>7.2f}")
    print(f"\nWorst CT shortfall over all cases: {worst_r[0]:.4f}"
          + (f" at q={worst_r[1][0]}, N={worst_r[1][1]}, eps={worst_r[1][2]:.2f}, "
             f"size {worst_r[1][3]:.4f}" if worst_r[1] else "")
          + f".  Largest eps at which the CT was ever below size: {worst_r_eps:.2f}")


def part_gamma(args):
    """Claim: the gamma false positive rate stays near alpha for large N."""
    rng = np.random.default_rng(args.seed)
    from scipy import stats
    print(f"moment-matched gamma, {args.draws} calibration draws, "
          f"{args.draws} independent test draws, alpha={args.alpha}")
    print(f"{'q':>26} {'n':>3} {'N':>7} {'FPR':>7}")
    worst = (0.0, None)
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(3, 12), weight_range=(1, 30))
        p0 = np.array(q, float) / sum(q)
        for N in (500, 5000, 50000):
            x = rng.multinomial(N, p0, size=args.draws)
            values = np.abs(np.diff(x - N * p0, axis=1)).sum(1)
            mean, var = values.mean(), values.var(ddof=1)
            scale = var / mean
            threshold = stats.gamma(a=mean / scale, scale=scale).isf(args.alpha)
            y = rng.multinomial(N, p0, size=args.draws)
            fpr = float(np.mean(np.abs(np.diff(y - N * p0, axis=1)).sum(1) > threshold))
            if abs(fpr - args.alpha) > worst[0]:
                worst = (abs(fpr - args.alpha), (q, N, fpr))
            if case < 4 or abs(fpr - args.alpha) > 0.006:
                print(f"{str(q):>26} {len(q):>3} {N:>7} {fpr:>7.4f}")
    print(f"\nWorst deviation from alpha: {worst[0]:.4f}"
          + (f" (q={worst[1][0]}, N={worst[1][1]}, FPR {worst[1][2]:.4f})" if worst[1] else ""))


def part_images(args):
    """Claim: the CT beats chi-square on real image code histograms."""
    import experiment_real_images as images
    from skimage import data as skdata
    names = [n for n in dir(skdata) if not n.startswith("_")]
    rng = np.random.default_rng(args.seed)
    print(f"all usable skimage images, n={args.n}, N={args.N}, eps=+-{args.eps}, "
          f"worse sign, {args.draws} draws")
    print(f"{'image':>16} {'min E':>7} {'CT':>7} {'X2':>7} {'X2p':>7} {'G':>7} {'ratio':>7}")
    ratios = []
    for name in names:
        try:
            image = images.load_gray_uint8(name)
        except Exception:
            continue
        if image.ndim != 2 or image.size < 20000:
            continue
        p0 = images.code_probabilities(image, args.n, 0.0)
        if p0.min() <= 0:
            continue
        x0 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
        s0 = images.statistics(x0, p0, args.N)
        cuts = {k: np.quantile(v, 1 - args.alpha) for k, v in s0.items()}
        worst = {}
        for sign in (+1, -1):
            p1 = images.code_probabilities(image, args.n, sign * args.eps)
            x1 = rng.multinomial(args.N, p1, size=args.draws).astype(float)
            s1 = images.statistics(x1, p1 * 0 + p0, args.N)
            for k in ("CT", "X2", "X2p", "G"):
                power = float(np.mean(s1[k] > cuts[k]))
                worst[k] = min(worst.get(k, 1.0), power)
        ratio = worst["CT"] / max(worst["X2p"], 1e-9)
        ratios.append((ratio, name))
        print(f"{name:>16} {args.N * p0.min():>7.2f} {worst['CT']:>7.3f} {worst['X2']:>7.3f} "
              f"{worst['X2p']:>7.3f} {worst['G']:>7.3f} {ratio:>7.2f}")
    ratios.sort()
    print(f"\n{len(ratios)} images.  CT / pooled-X2 ratio from {ratios[0][0]:.2f} "
          f"({ratios[0][1]}) to {ratios[-1][0]:.2f} ({ratios[-1][1]})")
    below = [r for r in ratios if r[0] < 1.0]
    print(f"images where the CT does NOT beat pooled chi-square: {len(below)}"
          + (": " + ", ".join(f"{n} ({v:.2f})" for v, n in below) if below else ""))


def part_altpath(args):
    """The two-weight path: sample from q1, centre on q0.  Every exact power in
    Table I comes from it, so it deserves the same brute-force treatment as the
    null-centred path."""
    rng = np.random.default_rng(args.seed)
    print("dr_law(N, q1, q0) vs brute force; mass must be (sum q1)^N")
    print(f"{'case':>5} {'q0':>20} {'q1':>20} {'N':>4} {'mass ok':>8} {'agrees':>7}")
    bad = 0
    for case in range(args.cases):
        n = int(rng.integers(2, 6))
        q0 = tuple(int(v) for v in rng.integers(1, 10, size=n))
        q1 = tuple(int(v) for v in rng.integers(1, 10, size=n))
        N = int(rng.integers(2, 10))
        law = dr_law(N, list(q1), list(q0))
        fast = {v: c for v, c in law if c}
        slow = brute_force(N, list(q1), list(q0))
        mass = sum(fast.values()) == sum(q1) ** N
        agree = fast == slow
        bad += not (mass and agree)
        if not (mass and agree) or case < 6:
            print(f"{case:>5} {str(q0):>20} {str(q1):>20} {N:>4} "
                  f"{'yes' if mass else 'NO':>8} {'yes' if agree else 'NO':>7}")
    print(f"\n{args.cases} random (q0, q1, N): {bad} disagreements with brute force")


def part_edge(args):
    """Degenerate inputs.  A silently wrong answer here would be a real defect."""
    print("degenerate and boundary inputs")
    bad = 0

    def report(label, condition, detail=""):
        nonlocal bad
        bad += not condition
        print(f"  {'OK  ' if condition else 'FAIL'}  {label}{(': ' + detail) if detail else ''}")

    report("n = 1, N = 5: single bin, no adjacent pair, S = 0 with all the mass",
           [(v, c) for v, c in dr_law(5, [3]) if c] == [(0, 3 ** 5)])
    report("n = 0: empty weight list returns an empty distribution",
           dr_law(4, []) == [])
    for N in (0, 1):
        law = [(v, c) for v, c in dr_law(N, [2, 3]) if c]
        report(f"n = 2, N = {N}: mass = L^N",
               sum(c for _, c in law) == 5 ** N, str(law))
    for q, N in (([4, 4, 4], 6), ([1, 1, 9], 6), ([9, 1, 1, 1, 1], 5),
                 ([2, 2, 5, 5], 7), ([1, 8], 9), ([7, 7], 8)):
        law = {v: c for v, c in dr_law(N, q) if c}
        report(f"q={q}, N={N}: DP == brute force and mass = L^N",
               law == brute_force(N, q) and sum(law.values()) == sum(q) ** N)
    for q, N in (([1, 2, 1], 4), ([1, 2, 1], 8), ([16, 8, 4, 2, 1, 1], 32),
                 ([16, 8, 4, 2, 1, 1], 33)):
        law = {v: c for v, c in dr_law(N, q) if c}
        report(f"q={q}, N={N} (N mod L = {N % sum(q)}): mass = L^N",
               sum(law.values()) == sum(q) ** N)
    for weight in (1, 2, 5):
        n, N = 3, 7
        law = {v: c for v, c in dr_law(N, [weight] * n) if c}
        plain = {v: c for v, c in du_law(N, [1] * n) if c}
        L = n * weight
        report(f"equal weights q_i = {weight} (L = {L}): S = L*DTV, counts x {weight}^N",
               law == {L * v: c * weight ** N for v, c in plain.items()})
    print(f"\n{bad} failures on degenerate inputs")


def part_levels(args):
    """The gamma approximation away from alpha = 0.05."""
    rng = np.random.default_rng(args.seed)
    from scipy import stats
    levels = (0.10, 0.05, 0.01, 0.005, 0.001)
    print(f"gamma FPR at several levels, {args.draws} calibration and {args.draws} "
          "independent test draws")
    header = "  ".join(f"a={a:<6}" for a in levels)
    print(f"{'q':>22} {'N':>7}  {header}")
    worst = (0.0, None)
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(3, 12), weight_range=(1, 30))
        p0 = np.array(q, float) / sum(q)
        for N in (500, 5000, 50000):
            x = rng.multinomial(N, p0, size=args.draws)
            values = np.abs(np.diff(x - N * p0, axis=1)).sum(1)
            mean, var = values.mean(), values.var(ddof=1)
            scale = var / mean
            law = stats.gamma(a=mean / scale, scale=scale)
            y = rng.multinomial(N, p0, size=args.draws)
            test = np.abs(np.diff(y - N * p0, axis=1)).sum(1)
            cells = []
            for a in levels:
                fpr = float(np.mean(test > law.isf(a)))
                cells.append(fpr)
                relative = abs(fpr - a) / a
                if relative > worst[0]:
                    worst = (relative, (q, N, a, fpr))
            print(f"{str(q):>22} {N:>7}  " + "  ".join(f"{v:<8.4f}" for v in cells))
    print(f"\nWorst RELATIVE deviation {worst[0]:.2f}x"
          + (f" at alpha={worst[1][2]}, q={worst[1][0]}, N={worst[1][1]}, "
             f"FPR {worst[1][3]:.4f}" if worst[1] else ""))


def part_misspec(args):
    """The Conclusions quote sizes for a null fitted from a calibration capture.

    experiment_variants.py part C computes them with ONE calibration draw per
    image, so the reported numbers average only five random estimates and are not
    stable to the seed (the audit of 2026-09-10 found 0.60/0.83 moving to
    0.68/0.91).  This part repeats the SAME construction -- calibration capture of
    M0 draws from the truth, the add-one fix when a code is unoccupied, critical
    value simulated under the estimate, test data from the truth -- with many
    calibration draws per image, and reports the mean and the spread.
    """
    import experiment_real_images as images
    rng = np.random.default_rng(args.seed)
    names = ["camera", "coins", "moon", "page", "cell"]
    truth = {name: images.code_probabilities(images.load_gray_uint8(name), args.n, 0.0)
             for name in names}
    print(f"size at alpha={args.alpha}; calibration capture of M0 draws from the truth, "
          f"add-one fix\nwhen a code is unoccupied; {args.repeats} calibration draws per "
          f"image, {args.draws} null and\n{args.draws} test draws each; mean over the "
          f"{5 * args.repeats} (image, draw) pairs with the range\n")
    print(f"{'M0/N':>5} " + " ".join(f"{k:>22}" for k in ("CT", "X2", "T(1)", "du")))
    keys = {"CT": "CT", "X2": "X2", "T(1)": "T(1)", "du": "DTV"}
    for ratio in (1, 5, 25, 125):
        M0 = ratio * args.N
        collected = {k: [] for k in keys}
        for name in names:
            p0 = truth[name]
            for _ in range(args.repeats):
                counts = rng.multinomial(M0, p0).astype(float)
                if counts.min() == 0:
                    counts = counts + 1.0            # the add-one fix of part C
                p_hat = counts / counts.sum()
                x0 = rng.multinomial(args.N, p_hat, size=args.draws).astype(float)
                x1 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
                s0 = images.statistics(x0, p_hat, args.N)
                s1 = images.statistics(x1, p_hat, args.N)
                for label, key in keys.items():
                    cut = np.quantile(s0[key], 1.0 - args.alpha)
                    collected[label].append(float(np.mean(s1[key] > cut)))
        line = f"{ratio:>5}"
        for label in ("CT", "X2", "T(1)", "du"):
            v = np.array(collected[label])
            line += f"   {v.mean():.3f} [{v.min():.3f},{v.max():.3f}]"
        print(line)
    print("\nThe Letter's Conclusions quote the M0/N = 1 and 125 rows.")


def part_indexn(args):
    """Does the index's ORDERING of the uncentered d_u's loss hold at every N?

    Two families are used, because the answer depends on the sample of nulls.
    SPANNING: geometric, Dirichlet, ramp and tent nulls chosen to cover the index
    axis, as experiment_rho_sweep.py does.  RANDOM: uniform random integer
    weights, which turn out to sit almost entirely at index ~ 1, so no ordering
    can be measured there -- a range restriction, not a counterexample, and a
    fact worth knowing in itself: a null with arbitrary weights is usually one on
    which the uncentered DTV would already have collapsed.
    """
    rng = np.random.default_rng(args.seed)

    def spanning():
        out = []
        for beta in (0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0):
            for n in (6, 10):
                w = np.exp(-beta * np.arange(n))
                out.append((f"geometric b={beta} n={n}", w / w.sum()))
        for a in (0.5, 2.0, 10.0):
            for _ in range(2):
                n = int(rng.integers(5, 14))
                w = rng.dirichlet(np.ones(n) * a)
                if w.min() > 1e-4:
                    out.append((f"Dirichlet a={a} n={n}", w))
        for n in (6, 10):
            w = np.arange(1.0, n + 1); out.append((f"ramp n={n}", w / w.sum()))
            w = np.minimum(np.arange(1.0, n + 1), n - np.arange(n))
            out.append((f"tent n={n}", w / w.sum()))
        return out

    def random_weights():
        out = []
        for _ in range(14):
            q = random_rational_null(rng, n_range=(4, 10), weight_range=(1, 20))
            if len(set(q)) > 1:
                out.append((f"random q={q}", np.array(q, float) / sum(q)))
        return out

    rank = lambda v: np.argsort(np.argsort(v))
    for label, family in (("SPANNING", spanning()), ("RANDOM WEIGHTS", random_weights())):
        print(f"\n{label}: {len(family)} nulls, l1={args.eps}, {args.draws} draws")
        print(f"{'N':>7} {'index range':>16} {'du worse sign':>16} {'CT worse sign':>16} "
              f"{'Spearman':>9}")
        for N in (500, 2000, 10000):
            rows = []
            for name, p0 in family:
                x0 = rng.multinomial(N, p0, size=args.draws).astype(float)
                r0 = x0 - N * p0
                dtv0 = np.abs(np.diff(x0, axis=1)).sum(1)
                res0 = np.abs(np.diff(r0, axis=1)).sum(1)
                linear0 = (np.sign(np.diff(p0)) * np.diff(x0, axis=1)).sum(1)
                index = float(np.corrcoef(dtv0, linear0)[0, 1]) if linear0.std() > 0 else 0.0
                cut_d = np.quantile(dtv0, 1 - args.alpha)
                cut_r = np.quantile(res0, 1 - args.alpha)
                worst_d = worst_r = 1.0
                for which in (0, 1):
                    p1 = comb_pair(p0, args.eps)[which]
                    x1 = rng.multinomial(N, p1, size=args.draws // 2).astype(float)
                    r1 = x1 - N * p0
                    worst_d = min(worst_d, float(np.mean(np.abs(np.diff(x1, axis=1)).sum(1) > cut_d)))
                    worst_r = min(worst_r, float(np.mean(np.abs(np.diff(r1, axis=1)).sum(1) > cut_r)))
                rows.append((index, worst_d, worst_r))
            idx = np.array([r[0] for r in rows])
            spearman = float(np.corrcoef(rank(idx), rank([-r[1] for r in rows]))[0, 1])
            print(f"{N:>7} {min(idx):>7.3f}..{max(idx):<8.3f} "
                  f"{min(r[1] for r in rows):>7.3f}..{max(r[1] for r in rows):<8.3f} "
                  f"{min(r[2] for r in rows):>7.3f}..{max(r[2] for r in rows):<8.3f} {spearman:>9.3f}")
        if label == "RANDOM WEIGHTS":
            share = float(np.mean([r[0] > 0.9 for r in rows]))
            print(f"  share of random-weight nulls with index above 0.9: {share:.0%}")


def part_cbound(args):
    """c <= min(L, 2^(n-1)), over random weights."""
    rng = np.random.default_rng(args.seed)
    print("c = number of residue classes of S modulo L on the support")
    print(f"{'q':>24} {'L':>5} {'N':>4} {'c':>4} {'bound':>6} {'ok':>4}")
    bad = 0
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(2, 7), weight_range=(1, 11))
        N = int(rng.integers(3, 12))
        L = sum(q)
        support = [v for v, c in dr_law(N, list(q)) if c]
        c = len({v % L for v in support})
        bound = min(L, 2 ** (len(q) - 1))
        ok = c <= bound
        bad += not ok
        if not ok or case < 10:
            print(f"{str(q):>24} {L:>5} {N:>4} {c:>4} {bound:>6} {'yes' if ok else 'NO':>4}")
    print(f"\n{args.cases} random cases: {bad} violations of c <= min(L, 2^(n-1))")


def part_printed(args):
    """Does the recurrence AS PRINTED in the Letter compute what the code computes?

    Every other check in this repository verifies the CODE.  A typo in the typeset
    equation would be invisible to all of them, and it is the equation a reader
    reimplements.  So this part transcribes the Letter's equation (4) and its
    single-bin initialization literally, in the pull form in which they are
    printed, evaluates them by memoized recursion, and compares with the shipped
    dynamic program and with brute force.

    As printed:
        single bin   C(1, M, M, 0) = binom(N, M) q_1^M,  else 0
        added bins   C(i, M, m, s) = binom(N-M+m, m) q_i^m
                                     * sum_{k=0}^{M-m} C(i-1, M-m, k,
                                           s - |L(m-k) - N(q_i - q_{i-1})|)
        with C = 0 whenever an argument leaves its range, and
        Pr[S = s] = sum_m C(n, N, m, s) / L^N.
    """
    from functools import lru_cache
    from math import comb as binomial
    rng = np.random.default_rng(args.seed)
    print("the Letter's equation (4), transcribed and evaluated as printed, against")
    print("the shipped DP and against brute force\n")
    print(f"{'case':>5} {'q':>22} {'N':>4} {'atoms':>6} {'= code':>7} {'= brute':>8} {'mass':>6}")
    bad = 0
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(2, 6), weight_range=(1, 8))
        n, L = len(q), sum(q)
        N = int(rng.integers(2, 9))

        @lru_cache(maxsize=None)
        def C(i, M, m, s):
            if i < 1 or M < 0 or m < 0 or m > M or M > N or s < 0:
                return 0
            if i == 1:
                return binomial(N, M) * q[0] ** M if m == M and s == 0 else 0
            total = 0
            for k in range(0, M - m + 1):
                increment = abs(L * (m - k) - N * (q[i - 1] - q[i - 2]))
                total += C(i - 1, M - m, k, s - increment)
            return binomial(N - M + m, m) * q[i - 1] ** m * total

        # Pr[S = s] = sum_m C(n, N, m, s) / L^N, over the attainable s
        shipped = {v: c for v, c in dr_law(N, list(q)) if c}
        printed = {}
        for s_value in sorted(shipped):
            mass = sum(C(n, N, m, s_value) for m in range(N + 1))
            if mass:
                printed[s_value] = mass
        # the printed form must not put mass anywhere the code does not, either
        extra = 0
        span = max(shipped) if shipped else 0
        for s_value in range(span + 1):
            if s_value in shipped:
                continue
            if sum(C(n, N, m, s_value) for m in range(N + 1)):
                extra += 1
        slow = brute_force(N, list(q))
        same_code = printed == shipped and extra == 0
        same_brute = printed == slow
        mass_ok = sum(printed.values()) == L ** N
        bad += not (same_code and same_brute and mass_ok)
        if not (same_code and same_brute and mass_ok) or case < 8:
            print(f"{case:>5} {str(q):>22} {N:>4} {len(printed):>6} "
                  f"{'yes' if same_code else 'NO':>7} {'yes' if same_brute else 'NO':>8} "
                  f"{'yes' if mass_ok else 'NO':>6}")
    print(f"\n{args.cases} random nulls: {bad} cases where the PRINTED recurrence "
          f"disagrees with the code or with brute force")


def part_pvalue(args):
    """Is the exact p-value actually a valid p-value?

    Section II-A defines the p-value of an observed s as Pr_H0[S >= s].  A valid
    p-value must be super-uniform under the null: Pr[P <= u] <= u for EVERY u.
    On a lattice this is where an off-by-one between >= and > would show, and no
    other check in the repository would catch it, because every other check looks
    at the distribution rather than at the inference built on top of it.  Here the
    law of P under H0 is computed exactly from the exact law of S, so the test is
    algebraic, not Monte Carlo.
    """
    rng = np.random.default_rng(args.seed)
    print("exact law of the p-value P = Pr[S >= s] under H0; a valid p-value needs")
    print("Pr[P <= u] <= u at every attainable u, and the test at level alpha needs")
    print("attained size <= alpha\n")
    print(f"{'q':>22} {'N':>4} {'atoms':>6} {'max Pr[P<=u]-u':>15} {'size(.05)':>10} {'ok':>4}")
    bad = 0
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(2, 6), weight_range=(1, 9))
        N = int(rng.integers(3, 13))
        L = sum(q)
        law = [(v, c) for v, c in dr_law(N, list(q)) if c]
        total = L ** N
        # P(s) = Pr[S >= s], evaluated at each atom s
        upper = {}
        running = 0
        for v, c in reversed(law):
            running += c
            upper[v] = F(running, total)
        worst = F(0)
        for v, c in law:
            u = upper[v]
            # Pr[P <= u] = mass on atoms whose p-value is <= u = atoms >= v
            mass = F(sum(cc for vv, cc in law if vv >= v), total)
            if mass - u > worst:
                worst = mass - u
        crit, size = tail(law, total, F(args.alpha).limit_denominator(10 ** 6))
        ok = worst <= 0 and float(size) <= args.alpha + 1e-12
        bad += not ok
        if not ok or case < 10:
            print(f"{str(q):>22} {N:>4} {len(law):>6} {float(worst):>15.2e} "
                  f"{float(size):>10.4f} {'yes' if ok else 'NO':>4}")
    print(f"\n{args.cases} random nulls: {bad} violations of super-uniformity or of "
          f"the attained size")


def part_local(args):
    """WHY the CT can be biased at small amplitude.

    d_r is a sum of absolute values of Delta r.  Under the alternative each term
    has mean N*eps*Delta delta and a standard deviation set by the ALTERNATIVE's
    own probabilities, and for a term with small mean E|Z| is governed by that
    standard deviation.  So E[d_r] picks up a first-order term in eps through
    the variance, which can be NEGATIVE, while the mean shift only helps at second
    order.  A negative first-order term means the statistic drifts down as the comb
    grows, which is power below size.

    Finite differences on a Monte Carlo mean are too noisy to estimate the second
    derivative, so instead the mean is traced on a grid of eps with many draws and
    the dip is read off directly: the smallest eps at which E[d_r] returns to its
    null value is the crossover the argument predicts.
    """
    rng = np.random.default_rng(args.seed)
    grid = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30])
    print("E[d_r] minus its null value, in units of its null standard deviation,")
    print(f"along the comb direction; {args.draws} draws per point\n")
    print(f"{'q':>20} {'N':>5} " + " ".join(f"{e:>7.2f}" for e in grid[1:]) + "   dips?")
    dips = 0
    for case in range(args.cases):
        q = random_rational_null(rng, n_range=(3, 7), weight_range=(1, 9))
        if len(set(q)) == 1:
            continue
        p0 = np.array(q, float) / sum(q)
        N = int(rng.choice([25, 100, 400]))
        def stats_at(eps):
            p1 = comb_pair(p0, eps)[0] if eps > 0 else p0
            x = rng.multinomial(N, p1, size=args.draws).astype(float)
            v = np.abs(np.diff(x - N * p0, axis=1)).sum(1)
            return float(v.mean()), float(v.std(ddof=1))
        base, sd = stats_at(0.0)
        shifts = [(stats_at(e)[0] - base) / sd for e in grid[1:]]
        negative = min(shifts) < -0.01
        dips += negative
        print(f"{str(q):>20} {N:>5} " + " ".join(f"{v:>7.3f}" for v in shifts)
              + ("   yes" if negative else "   no"))
    print(f"\n{dips} of {args.cases} nulls show E[d_r] BELOW its null value at some")
    print("small eps.  NOTE: a mean-based explanation of the bias was proposed here and")
    print("WITHDRAWN.  The exact curves in experiment_bias.py ('Mechanism, exactly') show")
    print("the mean moving under one percent in either direction at the worst amplitude")
    print("while the standard deviation falls one to three percent, so DISPERSION drives")
    print("it.  Nor is the bias confined to small eps: experiment_bias_exhaustive.py finds")
    print("shortfalls at every amplitude on the grid, worst at the largest.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=("exact", "phase", "bias", "gamma", "images",
                                 "altpath", "edge", "levels", "misspec",
                                 "indexn", "cbound", "local", "pvalue",
                                 "printed"))
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--draws", type=int, default=40000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    {"exact": part_exact, "phase": part_phase, "bias": part_bias,
     "gamma": part_gamma, "images": part_images, "altpath": part_altpath,
     "edge": part_edge, "levels": part_levels, "misspec": part_misspec,
     "indexn": part_indexn, "cbound": part_cbound,
     "local": part_local, "pvalue": part_pvalue,
     "printed": part_printed}[args.part](args)


if __name__ == "__main__":
    main()
