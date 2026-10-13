"""Is a gamma (or beta) approximation adequate for the null of the centered DTV d_r?

Section III of the parent Comb Test paper establishes that the uniform DTV null
is well approximated by a gamma once N leaves the exact regime, and that gamma is
preferred to the better-fitting beta because it is conservative in the tail.  The
centered statistic d_r must be re-examined, because subtracting N p_i changes the
shape: the floor at zero is retained, but it is now attained at x = N p, which is
not a lattice point unless L divides N q_i.

Everything here is computed from the exact integer distribution of L*d_r, so
the reported sizes are exact, not simulated.

Columns
  c_ex     smallest s with P(S > s) <= alpha, the exact critical value
  size_ex  the exact size actually achieved at c_ex (<= alpha by construction)
  c_ga     ceil of the moment-matched gamma quantile
  size_ga  exact P(S > c_ga), the true size of the gamma-based test
  c_be     ceil of the moment-matched beta quantile on [0, s_max]
  size_be  exact P(S > c_be)
  KS       sup |exact CDF - gamma CDF| over the real line (both limits at every atom)
"""

import argparse
from fractions import Fraction

import numpy as np
from scipy import stats

from du_exact import prepare_integer_weights
from dr_exact import calculate_exact_dr_distribution

FAMILIES = {
    "comb-shaped q=1:2:1:2:1 (n=5, L=7)": (1, 2, 1, 2, 1),
    "TPDF dither q=1:2:3:4:3:2:1 (n=7, L=16)": (1, 2, 3, 4, 3, 2, 1),
    "ramp q=1:2:3:4:5 (n=5, L=15)": (1, 2, 3, 4, 5),
    "run-length q=16:8:4:2:1:1 (n=6, L=32)": (16, 8, 4, 2, 1, 1),
    "uniform control (n=5, L=5)": (1, 1, 1, 1, 1),
}


def exact_mass(N, q):
    distribution = calculate_exact_dr_distribution(N, q)
    total = sum(q) ** N
    values = np.array([value for value, _ in distribution], dtype=float)
    mass = np.array([count / total for _, count in distribution], dtype=float)
    return values, mass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    alpha = args.alpha

    print(f"exact null of L*d_r, alpha={alpha}\n")
    header = (f"{'N':>5} {'c_ex':>7} {'size_ex':>8} {'c_ga':>7} {'size_ga':>8} "
              f"{'c_be':>7} {'size_be':>8} {'KS':>8}")
    for name, probs in FAMILIES.items():
        q = prepare_integer_weights(probs)
        print(name)
        print(header)
        for N in args.N:
            values, mass = exact_mass(N, q)
            survival = 1.0 - np.cumsum(mass)
            survival = np.clip(survival, 0.0, 1.0)
            index = int(np.argmax(survival <= alpha))
            c_ex, size_ex = values[index], survival[index]

            mean = float(values @ mass)
            variance = float(((values - mean) ** 2) @ mass)
            scale = variance / mean
            gamma = stats.gamma(a=mean / scale, scale=scale)
            c_ga = float(np.ceil(gamma.ppf(1.0 - alpha)))

            support = values[mass > 0]
            s_max = support.max()
            m, v = mean / s_max, variance / s_max ** 2
            common = m * (1.0 - m) / v - 1.0
            beta = stats.beta(a=m * common, b=(1.0 - m) * common, scale=s_max)
            c_be = float(np.ceil(beta.ppf(1.0 - alpha)))

            def exact_size(threshold):
                return float(mass[values > threshold].sum())

            exact_cdf = np.cumsum(mass)
            gamma_cdf = gamma.cdf(values)
            ks = float(max(np.max(np.abs(exact_cdf - gamma_cdf)),
                           np.max(np.abs(np.concatenate(([0.0], exact_cdf[:-1])) - gamma_cdf))))
            print(f"{N:>5} {c_ex:>7.0f} {size_ex:>8.4f} {c_ga:>7.0f} "
                  f"{exact_size(c_ga):>8.4f} {c_be:>7.0f} "
                  f"{exact_size(c_be):>8.4f} {ks:>8.4f}")
        print()


if __name__ == "__main__":
    main()
