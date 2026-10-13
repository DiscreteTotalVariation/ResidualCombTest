"""The collapse index of Table I's nulls at N = 40, computed EXACTLY.

The collapse index is Corr_H0(||x||_V, Lambda(x)) with ||x||_V = sum_i |D_i|,
Lambda = sum_i sign(d_i) D_i, D_i = x_i - x_{i-1}, d_i = p_i - p_{i-1}.  Every
moment it needs -- E|D_i|, E[|D_i||D_j|], E[|D_i| D_j], E[D_i D_j] -- involves at
most four bins, so each is a sum over the multinomial marginal of those bins
(the other bins pooled into one).  Enumerating those marginals in integer
arithmetic gives the index as a ratio of integers: no Monte Carlo, no seed.
The Letter's other collapse indices (Fig. 1b, Table II) are Monte Carlo at larger
N, where enumeration is out of reach; experiment_rho_diagnostic.py has the
estimator.  This script's numbers agree with that estimator to Monte Carlo
error (checked in verify_comb_test.py, group 5, for the ordering).

Output: results/experiment_exact_index.txt, read by tables/table1.py.

    python3 experiment_exact_index.py            # N = 10 20 30 40, Table I nulls
    python3 experiment_exact_index.py --N 40 --q 1 2 3 4 5
"""
import argparse
from fractions import Fraction
from math import comb, isqrt

NULLS = [("uniform control", (1, 1, 1, 1, 1)),
         ("comb null", (1, 2, 1, 2, 1)),
         ("ramp", (1, 2, 3, 4, 5)),
         ("tent", (1, 2, 3, 4, 3, 2, 1)),
         ("run-length", (16, 8, 4, 2, 1, 1))]


def compositions(total, parts):
    """All tuples of `parts` non-negative integers summing to at most `total`."""
    if parts == 0:
        yield ()
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def marginal_moments(N, q, bins, funcs):
    """sum over the marginal of `bins` (others pooled) of weight * f(counts) for
    each f in funcs; weights are integers with total L**N."""
    L = sum(q)
    rest_w = L - sum(q[b] for b in bins)
    k = len(bins)
    powers = [[q[b] ** m for m in range(N + 1)] for b in bins]
    rest_powers = [rest_w ** m for m in range(N + 1)]
    acc = [0] * len(funcs)
    for cnt in compositions(N, k):
        used = sum(cnt)
        rest = N - used
        # multinomial coefficient N! / (c_1! ... c_k! rest!) as a product of binomials
        w, left = 1, N
        for c in cnt:
            w *= comb(left, c)
            left -= c
        for b, c in enumerate(cnt):
            w *= powers[b][c]
        w *= rest_powers[rest]
        vals = dict(zip(bins, cnt))
        for j, f in enumerate(funcs):
            acc[j] += w * f(vals)
    return acc


def exact_index(N, q):
    """Fraction-valued numerator/denominator pieces and the index as a float."""
    n = len(q)
    L = sum(q)
    total = L ** N
    d = [q[i] - q[i - 1] for i in range(1, n)]              # slopes (times L), i = 2..n
    s = [(1 if v > 0 else -1 if v < 0 else 0) for v in d]     # sign(d_i), 0 where flat
    diffs = list(range(1, n))                                 # D_i uses bins i-1, i (0-based i)
    if all(v == 0 for v in s):
        return None, "Lambda is identically zero"
    E_abs = {}   # E|D_i| * total
    E_D = {}     # E[D_i] * total
    M_absabs, M_absD, M_DD = {}, {}, {}
    for i in diffs:
        (E_abs[i], E_D[i]) = marginal_moments(
            N, q, (i - 1, i),
            [lambda v, i=i: abs(v[i] - v[i - 1]), lambda v, i=i: v[i] - v[i - 1]])
    for a in diffs:
        for b in diffs:
            if b < a:
                continue
            bins = tuple(sorted({a - 1, a, b - 1, b}))
            fa = lambda v, a=a: v[a] - v[a - 1]
            fb = lambda v, b=b: v[b] - v[b - 1]
            r = marginal_moments(N, q, bins, [
                lambda v: abs(fa(v)) * abs(fb(v)),
                lambda v: abs(fa(v)) * fb(v),
                lambda v: fa(v) * abs(fb(v)),
                lambda v: fa(v) * fb(v)])
            M_absabs[(a, b)] = M_absabs[(b, a)] = r[0]
            M_absD[(a, b)] = r[1]          # E[|D_a| D_b]
            M_absD[(b, a)] = r[2]          # E[|D_b| D_a]
            M_DD[(a, b)] = M_DD[(b, a)] = r[3]
    T = Fraction(total)
    E_dtv = sum(E_abs.values()) / T
    E_lam = sum(s[i - 1] * E_D[i] for i in diffs) / T
    E_dtv2 = sum(M_absabs[(a, b)] for a in diffs for b in diffs) / T
    E_lam2 = sum(s[a - 1] * s[b - 1] * M_DD[(a, b)] for a in diffs for b in diffs) / T
    E_cross = sum(s[b - 1] * M_absD[(a, b)] for a in diffs for b in diffs) / T
    cov = E_cross - E_dtv * E_lam
    var_dtv = E_dtv2 - E_dtv ** 2
    var_lam = E_lam2 - E_lam ** 2
    if var_dtv == 0 or var_lam == 0:
        return None, "a variance is zero"
    # exact rational cov and variances; the square root is the only rounding
    return float(cov) / (float(var_dtv) * float(var_lam)) ** 0.5, \
        f"cov {float(cov):.6f}  var_dtv {float(var_dtv):.6f}  var_lambda {float(var_lam):.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--q", type=int, nargs="+", help="one null instead of Table I's five")
    a = ap.parse_args()
    nulls = [("custom", tuple(a.q))] if a.q else NULLS
    print("collapse index Corr_H0(||x||_V, Lambda) by exact enumeration of the <= 4-bin")
    print("multinomial marginals every moment needs; integer arithmetic, no Monte Carlo.")
    print("Lambda = sum_i sign(p_i - p_{i-1}) (x_i - x_{i-1}); a flat step has sign 0.")
    print("On the uniform null Lambda is identically zero and the index is undefined;")
    print("it is reported as 0, the convention of the Monte Carlo estimator.\n")
    for name, q in nulls:
        print(f"{name}   q={':'.join(map(str, q))}")
        print(f"{'N':>5} | {'index':>8} | moments")
        for N in a.N:
            idx, note = exact_index(N, q)
            shown = 0.0 if idx is None else idx
            print(f"{N:>5} | {shown:>8.4f} | {note}")
        print()


if __name__ == "__main__":
    main()
