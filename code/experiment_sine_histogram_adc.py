"""IEEE 1241 sine-wave histogram test with a non-uniform (arcsine) null.

The parent letter's ADC experiment assumed a stimulus that occupies all codes
uniformly, which IEEE Std 1241 permits but which in practice is approximated by a
slightly over-ranged sine wave whose code density is the ARCSINE law, not the
uniform one.  That density is steep at the rails and nearly flat in the middle,
so it is a natural non-uniform null: irrational (no exact DP), hence a test of the
gamma route, and U-shaped, hence a test of the linearity diagnostic on a null
whose slope varies across the record.

Null.  A sine of amplitude A slightly above full scale, uniformly random phase,
ideal n-code quantiser on [-1, 1).  With F(v) = 1/2 + asin(clip(v/A, -1, 1))/pi,
the code probabilities are p_k = F(e_{k+1}) - F(e_k) with e_k = -1 + 2k/n, the
end codes absorbing the clipped mass.  Every code has positive probability.

Defect (the parent's).  Successive-approximation LSB capacitor mismatch widens
every second code and narrows the rest, W_k = W (1 + (-1)^k eps); the edges move
accordingly and the code probabilities follow from the same F.  BOTH SIGNS of eps
are run: the sign is the phase of the physical defect, and a comb test must see
both equally.

Negative control (time-interleaving).  A two-way time-interleaved ADC with gain
mismatch 1+gamma and offset o on the second channel pools two smooth code
densities; the result is a SMOOTH departure, not a comb, so a histogram comb test
should have little power and chi-square should win.  This row is included so the
letter can say where the comb test (CT) is NOT the tool, in the parent's
manner.  (Interleaving mismatch shows up as spectral spurs, not as a histogram
comb; that is why the run-length application uses the two-lane analogy in the
TIME domain, where alternation does become a period-2 histogram effect.)

Everything is Monte-Carlo calibrated from the analytic null; a gamma fit to d_r
on an independent null batch is also reported (column `dr_gam`), which is the
procedure a practitioner would run at these N.
"""

import argparse

import numpy as np
from scipy import stats

EXPONENTS = (0.0, 0.5, 1.0)
ORDER = ["DTV", "DTV2", "CT", "T(1)", "min-p", "X2", "G", "AC", "vN",
         "X2asy", "dr_gam"]


def arcsine_cdf(v, amplitude):
    return 0.5 + np.arcsin(np.clip(v / amplitude, -1.0, 1.0)) / np.pi


def code_probabilities(edges, amplitude, gain=1.0, offset=0.0):
    """P(code k) for a quantiser with the given edges seeing gain*v + offset."""
    inner = (edges[1:-1] - offset) / gain
    cdf = arcsine_cdf(inner, amplitude)
    return np.diff(np.concatenate(([0.0], cdf, [1.0])))


def ideal_edges(n):
    return -1.0 + 2.0 * np.arange(n + 1) / n


def mismatched_edges(n, eps):
    widths = 1.0 + eps * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return -1.0 + 2.0 * np.concatenate(([0.0], np.cumsum(widths))) / widths.sum()


def statistics(x, p0, N):
    expected = N * p0
    centered = x - expected
    scale = np.sqrt(N * p0 * (1.0 - p0))
    alternating = np.where(np.arange(x.shape[1]) % 2 == 0, 1.0, -1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0) / expected), 0.0)
    out = {
        "DTV": np.abs(np.diff(x, axis=1)).sum(1),
        "X2": (centered ** 2 / expected).sum(1),
        "G": 2.0 * terms.sum(1),
        "AC": np.abs(centered @ alternating),
        "vN": (np.diff(centered, axis=1) ** 2).sum(1) / (centered ** 2).sum(1),
    }
    for t in EXPONENTS:
        out[f"T({t:g})"] = np.abs(np.diff(centered / scale ** t, axis=1)).sum(1)
    out["CT"] = out["T(0)"]           # d_r; T(1) is already keyed as "T(1)"
    return out


def survival_transform(reference):
    ordered = np.sort(reference)

    def transform(values):
        return 1.0 - np.searchsorted(ordered, values, side="left") / ordered.size
    return transform


def min_p(s, transforms):
    return np.min([transforms[f"T({t:g})"](s[f"T({t:g})"]) for t in EXPONENTS], axis=0)


def linearity(x, p0):
    differences = np.diff(x, axis=1)
    surrogate = (differences * np.sign(np.diff(p0))).sum(1)
    dtv = np.abs(differences).sum(1)
    return float(np.corrcoef(dtv, surrogate)[0, 1]) if surrogate.std() > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--amplitude", type=float, default=1.02,
                        help="sine amplitude relative to full scale (IEEE 1241 "
                             "recommends slight over-ranging)")
    parser.add_argument("--N", type=int, nargs="+", default=[500, 2000, 10000])
    parser.add_argument("--eps", type=float, nargs="+",
                        default=[0.0, 0.05, -0.05, 0.10, -0.10, 0.20, -0.20])
    parser.add_argument("--ti_gain", type=float, default=0.02,
                        help="gain mismatch of the second interleaved channel")
    parser.add_argument("--ti_offset_lsb", type=float, default=0.3,
                        help="offset of the second interleaved channel, in LSB")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null_draws", type=int, default=40000)
    parser.add_argument("--alt_draws", type=int, default=20000)
    parser.add_argument("--check_samples", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n, A = args.n, args.amplitude
    edges = ideal_edges(n)
    p0 = code_probabilities(edges, A)
    lsb = 2.0 / n

    print(f"IEEE 1241 sine-wave histogram test, n={n} codes, amplitude {A} FS, "
          f"alpha={args.alpha}")
    print(f"null = arcsine code density (irrational): p_min={p0.min():.5f} "
          f"(centre), p_max={p0.max():.5f} (rails), ratio {p0.max()/p0.min():.1f}")

    # sanity: the analytic null against actual random-phase sine samples
    phase = rng.uniform(0.0, 2.0 * np.pi, args.check_samples)
    codes = np.clip(np.floor((A * np.sin(phase) + 1.0) / lsb), 0, n - 1).astype(int)
    empirical = np.bincount(codes, minlength=n) / args.check_samples
    print(f"analytic vs simulated null, {args.check_samples:,} sine samples: "
          f"l1 gap {np.abs(empirical - p0).sum():.5f}, "
          f"chi2 p={stats.chisquare(np.bincount(codes, minlength=n), args.check_samples * p0).pvalue:.3f}\n")

    alternatives = {}
    for eps in args.eps:
        alternatives[("comb", eps)] = code_probabilities(mismatched_edges(n, eps), A)
    p_ti = 0.5 * p0 + 0.5 * code_probabilities(edges, A, gain=1.0 + args.ti_gain,
                                                 offset=args.ti_offset_lsb * lsb)
    alternatives[("TI", 0.0)] = p_ti
    print("l1 distances from the null:")
    for key, p1 in alternatives.items():
        if key[1] != 0.0 or key[0] == "TI":
            print(f"  {key[0]:>4} eps={key[1]:>6.2f}   l1={np.abs(p1 - p0).sum():.4f}")
    print()

    for N in args.N:
        x0 = rng.multinomial(N, p0, size=args.null_draws).astype(float)
        x0b = rng.multinomial(N, p0, size=args.null_draws).astype(float)
        s0, s0b = statistics(x0, p0, N), statistics(x0b, p0, N)
        transforms = {f"T({t:g})": survival_transform(s0[f"T({t:g})"]) for t in EXPONENTS}
        cuts = {k: np.quantile(s0[k], 1.0 - args.alpha) for k in s0}
        cut_minp = np.quantile(min_p(s0b, transforms), args.alpha)
        low2, high2 = np.quantile(s0["DTV"], [args.alpha / 2, 1.0 - args.alpha / 2])
        x2_asy = stats.chi2.isf(args.alpha, n - 1)
        shape, scale = (s0["CT"].mean() ** 2 / s0["CT"].var(),
                        s0["CT"].var() / s0["CT"].mean())
        gamma_cut = stats.gamma.isf(args.alpha, shape, scale=scale)
        lin = linearity(x0, p0)
        print(f"N = {N}   (mean count per code {N / n:.1f}; linearity of the "
              f"arcsine null {lin:.3f})")
        print(f"{'kind':>5} {'eps':>6} " + " ".join(f"{k:>9}" for k in ORDER))

        def row(kind, eps, p1, tag=""):
            x1 = rng.multinomial(N, p1, size=args.alt_draws).astype(float)
            s1 = statistics(x1, p0, N)
            r = {k: float(np.mean(s1[k] > cuts[k])) for k in cuts if k in ORDER}
            r["DTV2"] = float(np.mean((s1["DTV"] > high2) | (s1["DTV"] < low2)))
            r["min-p"] = float(np.mean(min_p(s1, transforms) < cut_minp))
            r["X2asy"] = float(np.mean(s1["X2"] > x2_asy))
            r["dr_gam"] = float(np.mean(s1["CT"] > gamma_cut))
            print(f"{kind:>5} {eps:>6.2f} " + " ".join(f"{r[k]:>9.3f}" for k in ORDER) + tag)

        for eps in args.eps:
            row("comb", eps, alternatives[("comb", eps)], "  (size)" if eps == 0.0 else "")
        row("TI", 0.0, p_ti, "  (gain/offset mismatch: smooth, not a comb)")
        print()


if __name__ == "__main__":
    main()
