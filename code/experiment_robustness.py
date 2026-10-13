"""Six further experiments, none of which duplicates existing work.

  --part local     a LOCALIZED comb: a real converter defect affects some codes,
                   not all.  The DTV is built from adjacent differences, so a
                   comb confined to part of the range may suit it better than a
                   global one, or worse.  Nothing in the Letter tests this.
  --part drift     the null DRIFTS between calibration and test (a systematic
                   change, unlike the random estimation error already studied).
  --part samplesize power against N at fixed departure, and the N at which the CT
                   overtakes chi-square: the practitioner's question.
  --part gammaexact the gamma critical value against the EXACT one at alpha =
                   0.05, 0.01, 0.001.  Stronger than comparing gamma with Monte
                   Carlo of the same law, because the exact null is ground truth.
  --part twosided  the Letter argues the two-sided variant is unnecessary for the
                   centered DTV d_r.  Quantify what it would cost if used.
  --part discrete  how much level the exact test loses to discreteness, and what a
                   randomized test would recover.
"""
import argparse
from fractions import Fraction as F
import numpy as np
from dr_exact import calculate_exact_dr_distribution as dr_law
from du_exact import calculate_exact_dtv_distribution_nonuniform as du_law

NULLS = [("tent 1:2:3:4:3:2:1", np.array([1., 2, 3, 4, 3, 2, 1])),
         ("ramp 1:2:3:4:5:6", np.arange(1., 7)),
         ("run-length 16:8:4:2:1:1", np.array([16., 8, 4, 2, 1, 1])),
         ("geometric n=12 b=0.2", np.exp(-0.2 * np.arange(12)))]


def tail(d, total, alpha):
    s = total
    for v, c in d:
        s -= c
        if F(s, total) <= alpha:
            return v, F(s, total)
    return d[-1][0], F(0)


def scale_to(p0, delta, distance):
    """Project the direction onto the simplex's tangent space (so that it sums to
    zero and p0 +- delta stays a probability vector), then scale it to l1
    `distance`.  Omitting the projection leaves p0 + delta off the simplex, which
    is what an earlier version of this file did."""
    delta = delta - delta.sum() * p0
    s = np.abs(delta).sum()
    if s < 1e-12:
        return None
    delta = delta * distance / s
    if (p0 + delta).min() <= 0 or (p0 - delta).min() <= 0:
        live = np.abs(delta) > 0
        room = np.min(p0[live] / np.abs(delta)[live])
        delta = delta * room * 0.999
    return delta


def powers(rng, p0, delta, N, draws, alpha):
    for sign in (+1, -1):
        p1 = p0 + sign * delta
        assert abs(p1.sum() - 1.0) < 1e-12 and p1.min() > 0, "alternative off the simplex"
    x0 = rng.multinomial(N, p0, size=draws).astype(float)
    r0 = x0 - N * p0
    signs = (-1.0) ** np.arange(len(p0))
    stats0 = {"CT": np.abs(np.diff(r0, axis=1)).sum(1),
              "du": np.abs(np.diff(x0, axis=1)).sum(1),
              "X2": (r0 ** 2 / (N * p0)).sum(1),
              "AC": np.abs(r0 @ signs)}
    cuts = {k: np.quantile(v, 1 - alpha) for k, v in stats0.items()}
    worst = {k: 1.0 for k in stats0}
    for sign in (+1, -1):
        x1 = rng.multinomial(N, p0 + sign * delta, size=draws // 2).astype(float)
        r1 = x1 - N * p0
        s1 = {"CT": np.abs(np.diff(r1, axis=1)).sum(1),
              "du": np.abs(np.diff(x1, axis=1)).sum(1),
              "X2": (r1 ** 2 / (N * p0)).sum(1),
              "AC": np.abs(r1 @ signs)}
        for k in worst:
            worst[k] = min(worst[k], float(np.mean(s1[k] > cuts[k])))
    return worst


def part_local(args):
    rng = np.random.default_rng(args.seed)
    print(f"comb confined to a FRACTION of the bins, same l1={args.distance}, "
          f"N={args.N}, worse sign\n")
    print(f"{'null':>22} {'fraction':>9} {'bins':>5} " + " ".join(f"{k:>7}" for k in ("CT", "X2", "AC", "du")))
    for name, weights in NULLS:
        p0 = weights / weights.sum(); n = len(p0)
        for fraction in (0.25, 0.5, 1.0):
            width = max(2, int(round(fraction * n)))
            start = (n - width) // 2
            mask = np.zeros(n); mask[start:start + width] = 1.0
            delta = scale_to(p0, p0 * ((-1.0) ** np.arange(n)) * mask, args.distance)
            if delta is None:
                continue
            w = powers(rng, p0, delta, args.N, args.draws, args.alpha)
            print(f"{name:>22} {fraction:>9.2f} {width:>5} "
                  + " ".join(f"{w[k]:>7.3f}" for k in ("CT", "X2", "AC", "du")))
        print()


def part_drift(args):
    rng = np.random.default_rng(args.seed)
    print("the null used for calibration is p0; the data come from a DRIFTED null")
    print(f"p0 tilted by a relative gradient d, l1 distance shown; N={args.N}, "
          f"nominal alpha={args.alpha}\n")
    print(f"{'null':>22} {'drift l1':>9} " + " ".join(f"{k:>7}" for k in ("CT", "X2", "AC", "du")))
    for name, weights in NULLS:
        p0 = weights / weights.sum(); n = len(p0)
        x0 = rng.multinomial(args.N, p0, size=args.draws).astype(float)
        r0 = x0 - args.N * p0
        signs = (-1.0) ** np.arange(n)
        cuts = {"CT": np.quantile(np.abs(np.diff(r0, axis=1)).sum(1), 1 - args.alpha),
                "du": np.quantile(np.abs(np.diff(x0, axis=1)).sum(1), 1 - args.alpha),
                "X2": np.quantile((r0 ** 2 / (args.N * p0)).sum(1), 1 - args.alpha),
                "AC": np.quantile(np.abs(r0 @ signs), 1 - args.alpha)}
        for d in (0.0, 0.01, 0.02, 0.05, 0.10):
            tilt = np.linspace(-1, 1, n)
            p1 = p0 * (1 + d * tilt)
            p1 = p1 / p1.sum()
            if p1.min() <= 0:
                continue
            x1 = rng.multinomial(args.N, p1, size=args.draws).astype(float)
            r1 = x1 - args.N * p0
            rates = {"CT": float(np.mean(np.abs(np.diff(r1, axis=1)).sum(1) > cuts["CT"])),
                     "du": float(np.mean(np.abs(np.diff(x1, axis=1)).sum(1) > cuts["du"])),
                     "X2": float(np.mean((r1 ** 2 / (args.N * p0)).sum(1) > cuts["X2"])),
                     "AC": float(np.mean(np.abs(r1 @ signs) > cuts["AC"]))}
            print(f"{name:>22} {float(np.abs(p1 - p0).sum()):>9.4f} "
                  + " ".join(f"{rates[k]:>7.3f}" for k in ("CT", "X2", "AC", "du")))
        print()


def part_samplesize(args):
    rng = np.random.default_rng(args.seed)
    print(f"power against N at fixed l1={args.distance}, period-2 comb, worse sign\n")
    print(f"{'null':>22} {'N':>7} " + " ".join(f"{k:>7}" for k in ("CT", "X2", "AC", "du")) + "    CT/X2")
    for name, weights in NULLS:
        p0 = weights / weights.sum(); n = len(p0)
        delta = scale_to(p0, p0 * ((-1.0) ** np.arange(n)), args.distance)
        crossover = None
        for N in (100, 200, 500, 1000, 2000, 5000):
            w = powers(rng, p0, delta, N, args.draws, args.alpha)
            ratio = w["CT"] / max(w["X2"], 1e-9)
            if crossover is None and w["CT"] > w["X2"]:
                crossover = N
            print(f"{name:>22} {N:>7} " + " ".join(f"{w[k]:>7.3f}" for k in ("CT", "X2", "AC", "du"))
                  + f"   {ratio:>6.2f}")
        print(f"{'':>22} CT overtakes X2 from N = {crossover}\n")


def part_gammaexact(args):
    from scipy import stats
    print("gamma critical value against the EXACT one, and the exact size the gamma")
    print("cut attains; the exact null is ground truth here, not Monte Carlo\n")
    print(f"{'q':>22} {'N':>4} {'alpha':>7} {'exact cut':>10} {'gamma cut':>10} "
          f"{'exact size at gamma cut':>24}")
    for q, N in (((1, 2, 1, 2, 1), 40), ((1, 2, 1, 2, 1), 100), ((16, 8, 4, 2, 1, 1), 40),
                 ((1, 2, 3, 4, 3, 2, 1), 40), ((1, 2, 3, 4, 5, 6), 60)):
        law = dr_law(N, list(q))
        total = sum(q) ** N
        values = np.array([v for v, c in law], dtype=float)
        mass = np.array([float(F(c, total)) for v, c in law])
        mean = float(values @ mass)
        var = float(((values - mean) ** 2) @ mass)
        gamma = stats.gamma(a=mean * mean / var, scale=var / mean)
        for alpha in (0.05, 0.01, 0.001):
            cut_exact, size_exact = tail(law, total, F(alpha).limit_denominator(10 ** 9))
            cut_gamma = float(np.ceil(gamma.isf(alpha)))
            attained = float(sum(F(c, total) for v, c in law if v > cut_gamma))
            print(f"{str(q):>22} {N:>4} {alpha:>7.3f} {cut_exact:>10} {cut_gamma:>10.0f} "
                  f"{attained:>24.5f}")
        print()


def part_twosided(args):
    print("what a two-sided CT would cost.  The Letter argues it is")
    print("unnecessary because the statistic's minimum is the expected histogram;")
    print("here the equal-tailed two-sided version is computed exactly alongside.\n")
    print(f"{'q':>22} {'N':>4} {'one-sided size':>15} {'power':>8} | "
          f"{'two-sided size':>15} {'power':>8} {'loss':>7}")
    for q0, N, num, den in (((1, 2, 1, 2, 1), 40, 1, 5), ((16, 8, 4, 2, 1, 1), 20, 1, 5),
                            ((1, 2, 3, 4, 3, 2, 1), 16, 1, 5), ((1, 2, 3, 4, 5, 6), 20, 1, 5)):
        L0 = sum(q0)
        law = dr_law(N, list(q0))
        total = L0 ** N
        alpha = F(1, 20)
        cut, size = tail(law, total, alpha)
        # equal-tailed: alpha/2 in each tail
        low, mass_low = None, F(0)
        for v, c in law:
            if mass_low + F(c, total) > alpha / 2:
                break
            if c:
                mass_low += F(c, total); low = v
        high, mass_high = tail(law, total, alpha / 2)
        size_two = mass_low + mass_high
        q1 = tuple(v * (den + (1 if i % 2 == 0 else -1) * num) for i, v in enumerate(q0))
        L1 = sum(q1)
        alt = dr_law(N, list(q1), list(q0))
        p_one = sum(F(c, L1 ** N) for v, c in alt if v > cut)
        p_two = sum(F(c, L1 ** N) for v, c in alt if v > high or (low is not None and v <= low))
        print(f"{str(q0):>22} {N:>4} {float(size):>15.4f} {float(p_one):>8.4f} | "
              f"{float(size_two):>15.4f} {float(p_two):>8.4f} {float(p_one - p_two):>+7.4f}")
    print("\nA positive loss means the two-sided version is strictly worse against the")
    print("comb, which is the Letter's point stated as a number rather than an argument.")


def part_discrete(args):
    print("level lost to discreteness: the exact test's attained size against the")
    print("nominal 0.05, and what a randomized test would recover\n")
    print(f"{'q':>22} {'N':>5} {'L':>5} {'atoms':>6} {'attained':>9} {'gap':>8} "
          f"{'atom at the cut':>16}")
    for q, N in (((1, 2, 1, 2, 1), 10), ((1, 2, 1, 2, 1), 40), ((1, 2, 1, 2, 1), 100),
                 ((16, 8, 4, 2, 1, 1), 20), ((16, 8, 4, 2, 1, 1), 60),
                 ((1, 2, 3, 4, 3, 2, 1), 20), ((1, 2, 3, 4, 3, 2, 1), 60)):
        law = [(v, c) for v, c in dr_law(N, list(q)) if c]
        total = sum(q) ** N
        cut, size = tail(law, total, F(1, 20))
        atom = next((F(c, total) for v, c in law if v == cut), F(0))
        print(f"{str(q):>22} {N:>5} {sum(q):>5} {len(law):>6} {float(size):>9.5f} "
              f"{0.05 - float(size):>8.5f} {float(atom):>16.5f}")
    print("\nThe gap is level the test cannot use; a randomized test would recover it by")
    print("rejecting at the boundary atom with the right probability.  The gap shrinks")
    print("as the support grows, which is why it matters only at small N.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True,
                        choices=("local", "drift", "samplesize", "gammaexact",
                                 "twosided", "discrete"))
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=40000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--distance", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    {"local": part_local, "drift": part_drift, "samplesize": part_samplesize,
     "gammaexact": part_gammaexact, "twosided": part_twosided,
     "discrete": part_discrete}[args.part](args)


if __name__ == "__main__":
    main()
