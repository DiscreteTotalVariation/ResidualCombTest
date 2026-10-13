"""Two numerical experiments motivating the centered DTV d_r (the comb test, CT)
and its standardized variant T(1) for a non-uniform null.

Part 1 (telescoping): under a non-uniform null the absolute values in
    d_u(x) = sum_i |x_i - x_{i-1}|   (the uncentered DTV)
stop folding once N|p_i - p_{i-1}| dominates sqrt(N(p_i + p_{i-1})). d_u then
degenerates into the fixed linear functional
    T_lin(x) = sum_i s_i (x_i - x_{i-1}),   s_i = sign(p_i - p_{i-1}),
which telescopes and depends only on a few bins. We measure the degeneracy
directly (fold rate, correlation with T_lin) and its cost in power.

Part 2 (envelope): a comb perturbation can be additive (p_i = p_i^0 + s_i*delta,
constant amplitude) or multiplicative (p_i = p_i^0 (1 + s_i*eps), amplitude
following the baseline envelope). We compare
    CT   = d_r  = sum_i |e_i - e_{i-1}|,        e_i = x_i - N p_i (centered counts)
    T(1)        = sum_i |z_i - z_{i-1}|,        z_i = e_i / sqrt(N p_i (1-p_i))
under both, at matched L1 distance from the null.

All critical values are Monte-Carlo calibrated from the null, so every test is
compared at the same actual size.
"""

import numpy as np

ALPHA = 0.05
B_NULL = 100000
B_ALT = 40000

NULLS = {
    # Run-length histogram of a fair i.i.d. bitstream, tail pooled at bin 6.
    # P(run = k) = 2^-k, so q = (16, 8, 4, 2, 1, 1) over L = 32. Monotone.
    "run-length (geometric, q=16:8:4:2:1:1, L=32)": np.array(
        [16, 8, 4, 2, 1, 1], dtype=float),
    # TPDF dither / triangular: convolution of two discrete uniforms.
    "TPDF dither (tent, q=1:2:3:4:3:2:1, L=16)": np.array(
        [1, 2, 3, 4, 3, 2, 1], dtype=float),
    # Ramp / triangular DAC stimulus.
    "ramp (q=1:2:3:4:5:6, L=21)": np.array([1, 2, 3, 4, 5, 6], dtype=float),
    # Control: the uniform null of the original Comb Test.
    "uniform (control, n=6)": np.ones(6),
}

NS = [50, 200, 1000, 5000, 20000]
L1_DISTANCE = 0.05


def comb_alternative(p0, kind, l1_distance):
    """Comb perturbation of p0, rescaled to a given L1 distance."""
    n = len(p0)
    signs = np.array([(-1.0) ** i for i in range(n)])
    amplitude = p0 if kind == "multiplicative" else np.ones(n) / n
    delta = signs * amplitude
    delta = delta - delta.sum() * p0          # keep sum(p) = 1
    delta = delta * l1_distance / np.abs(delta).sum()
    p1 = p0 + delta
    assert p1.min() > 0 and abs(p1.sum() - 1) < 1e-12
    return p1


def statistics(x, p0, N):
    """All four test statistics for a batch of histograms x (B, n)."""
    expected = N * p0
    centered = x - expected
    z = centered / np.sqrt(N * p0 * (1 - p0))
    return {
        "du": np.abs(np.diff(x, axis=1)).sum(1),
        "CT": np.abs(np.diff(centered, axis=1)).sum(1),
        "T(1)": np.abs(np.diff(z, axis=1)).sum(1),
        "chi2": (centered ** 2 / expected).sum(1),
    }


def power_at(null_values, alt_values, alpha=ALPHA, two_sided=False):
    if two_sided:
        lo, hi = np.quantile(null_values, [alpha / 2, 1 - alpha / 2])
        return float(np.mean((alt_values > hi) | (alt_values < lo)))
    threshold = np.quantile(null_values, 1 - alpha)
    return float(np.mean(alt_values > threshold))


def part1_telescoping(rng):
    print("=" * 78)
    print("PART 1  Degeneracy of the uncentered DTV d_u under a non-uniform null")
    print("=" * 78)
    print("fold rate = P(sign(x_i - x_{i-1}) != sign(p_i - p_{i-1})), averaged over i")
    print("corr      = corr(d_u, T_lin), T_lin = sum_i s_i (x_i - x_{i-1})")
    print("#eff bins = number of bins T_lin actually depends on (after telescoping)")
    print()
    for name, q in NULLS.items():
        p0 = q / q.sum()
        signs = np.sign(np.diff(p0))
        # T_lin = sum_i s_i (x_i - x_{i-1}) collapses to a weighted sum of x.
        coefficients = np.zeros(len(p0))
        coefficients[1:] += signs
        coefficients[:-1] -= signs
        effective = int(np.count_nonzero(coefficients))
        print(f"{name}   [T_lin coefficients {coefficients.astype(int)}, "
              f"#eff bins {effective}/{len(p0)}]")
        for N in NS:
            x = rng.multinomial(N, p0, size=20000).astype(float)
            differences = np.diff(x, axis=1)
            fold_rate = float(np.mean(np.sign(differences) != signs))
            dtv = np.abs(differences).sum(1)
            t_lin = (differences * signs).sum(1)
            correlation = float(np.corrcoef(dtv, t_lin)[0, 1])
            print(f"    N={N:6d}   fold rate {fold_rate:6.4f}   corr {correlation:7.4f}")
        print()


def part2_power(rng):
    print("=" * 78)
    print("PART 2  Power at matched L1 distance "
          f"({L1_DISTANCE}), alpha={ALPHA}, MC-calibrated critical values")
    print("=" * 78)
    for name, q in NULLS.items():
        p0 = q / q.sum()
        print(name)
        for kind in ("additive", "multiplicative"):
            p1 = comb_alternative(p0, kind, L1_DISTANCE)
            print(f"  {kind} comb   p1 = {np.array2string(p1, precision=4)}")
            header = (f"    {'N':>6} {'du':>9} {'du2s':>10} "
                      f"{'CT':>8} {'T(1)':>8} {'chi2':>8}")
            print(header)
            for N in NS:
                x0 = rng.multinomial(N, p0, size=B_NULL).astype(float)
                x1 = rng.multinomial(N, p1, size=B_ALT).astype(float)
                s0 = statistics(x0, p0, N)
                s1 = statistics(x1, p0, N)
                du = power_at(s0["du"], s1["du"])
                du2 = power_at(s0["du"], s1["du"], two_sided=True)
                ct = power_at(s0["CT"], s1["CT"])
                std = power_at(s0["T(1)"], s1["T(1)"])
                chi = power_at(s0["chi2"], s1["chi2"])
                print(f"    {N:6d} {du:9.3f} {du2:10.3f} "
                      f"{ct:8.3f} {std:8.3f} {chi:8.3f}")
            print()


def main():
    rng = np.random.default_rng(31)
    part1_telescoping(rng)
    part2_power(rng)


if __name__ == "__main__":
    main()
