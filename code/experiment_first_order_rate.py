"""First-order (local) rejection rate along a comb direction, computed exactly.

Claim checked (Letter, Sec. III).  For x ~ Mult(N, p) and alternatives
p + a*delta (sum delta = 0) the rejection rate of an upper-tail rule T > c is a
polynomial in a, and its derivative at a = 0 is

    beta'(0) = Cov_H0( 1{T > c},  sum_j delta_j x_j / p_j ),

because d/da log P_a(x) at a = 0 is the score sum_j delta_j x_j / p_j, whose null
mean is N sum_j delta_j = 0.  Three statistics are compared:

    d_r     = sum_i |(x_{i+1} - x_i) - N (p_{i+1} - p_i)|   the comb test (CT), as S = L d_r
    d_u     = sum_i |x_{i+1} - x_i|                         the uncentered DTV
    Lambda  = sum_i sgn(g_i) (x_{i+1} - x_i),  g_i = p_{i+1} - p_i,
              the linear functional d_u locks to on a sloped null.

d_r is even in the centered counts e = x - N p and the score is odd in them, so
its slope is nonzero only through the multinomial's asymmetry (skewness and the
lattice); Lambda is linear and its slope grows like sqrt(N).  The claim under
test: slope(d_r) / slope(Lambda) = O(N^(-1/2)), i.e. slope(d_r)/sqrt(N) halves
in log scale at rate 1/2 per log N while slope(Lambda)/sqrt(N) stays flat.

delta is the Letter's multiplicative period-2 comb (experiment_exact_power.py,
comb_pair): step_i = q_i (s_i L - sum_k s_k q_k), s_i = +1 on odd bins (1-based),
so '+' raises the odd bins; delta = step / sum|step| has unit l1 norm.  The slope
is per unit l1 amplitude.

Method.  No Monte Carlo.  Every multinomial outcome inside a box of
+-K conditional standard deviations (+3) around the conditional means is
enumerated (numpy, float64 log-pmf); the mass outside the box is reported and
is printed per row ('mass out').  For N <= --int_N the same quantities are ALSO computed by full
enumeration in exact integer arithmetic (no floats at all) and the two are
compared.  For one statistic, all values v get P(T = v) and E[1{T = v} score],
so both the integer cut c (smallest c with P(T > c) <= alpha) and the randomized
cut (reject T > c, and T = c with probability gamma, size exactly alpha) are read
off the same pass.

The exact two-sign power gap of d_r at a fixed amplitude is computed with the
exact DP of dr_exact.py (alternative-centred path) and compared with the linear
prediction 2 a beta'(0); at the Table I distance it must reproduce
results/experiment_exact_power.txt digit for digit.

Output: results/experiment_first_order_rate.txt (no seed: nothing is random).

    python3 experiment_first_order_rate.py
    python3 experiment_first_order_rate.py --N 20 40 80 160 --jobs 8
"""
import argparse
import os
import platform
import sys
import time
from fractions import Fraction
from math import comb, isqrt, lgamma, log, sqrt
from multiprocessing import Pool

import numpy as np
import scipy

from experiment_exact_power import comb_pair, exceedance, tail
from dr_exact import calculate_exact_dr_distribution

NULLS = [("ramp", (1, 2, 3, 4, 5)),
         ("run-length", (16, 8, 4, 2, 1, 1)),
         ("comb null", (1, 2, 1, 2, 1))]
STATS = ("d_r", "d_u", "Lambda")


def sgn(v):
    return (v > 0) - (v < 0)


def comb_direction(q):
    """Integer score weights W and the factor turning E[1{.} W.x] into the
    derivative along the unit-l1 comb: delta_j / p_j = L W_j / A."""
    L = sum(q)
    s = [1 if i % 2 == 0 else -1 for i in range(len(q))]
    imbalance = sum(si * qi for si, qi in zip(s, q))
    W = [si * L - imbalance for si in s]
    step = [qi * w for qi, w in zip(q, W)]
    A = sum(abs(v) for v in step)
    assert sum(step) == 0
    return W, Fraction(L, A), [Fraction(v, A) for v in step]


def statistic_offsets(N, q):
    """Value index offset and length per statistic (all are integers)."""
    n, L = len(q), sum(q)
    slope = sum(abs(q[i + 1] - q[i]) for i in range(n - 1))
    up_r = 2 * N * L + N * slope
    return {"d_r": (0, up_r + 1), "d_u": (0, 2 * N + 1),
            "Lambda": (2 * N, 4 * N + 1)}


# ---------------------------------------------------------------- float pass

def _cond_range(R, p, K):
    if p >= 1.0:
        return R, R
    m = R * p
    sd = sqrt(R * p * (1.0 - p))
    return max(0, int(m - K * sd) - 3), min(R, int(m + K * sd) + 4)


def _worker(task):
    """Enumerate every outcome with x_1 = x1 inside the box; return per-value
    sums of pmf and pmf*score for the three statistics, and the mass seen."""
    N, q, K, x1 = task
    n, L = len(q), sum(q)
    p = [v / L for v in q]
    tails = [sum(p[i:]) for i in range(n)]
    lg = np.array([lgamma(k + 1) for k in range(N + 1)])
    logp = [log(v) for v in p]
    W = comb_direction(q)[0]
    g = [q[i + 1] - q[i] for i in range(n - 1)]
    sg = [sgn(v) for v in g]
    offs = statistic_offsets(N, q)
    acc = {s: [np.zeros(offs[s][1]), np.zeros(offs[s][1])] for s in STATS}
    mass = 0.0
    const = lgamma(N + 1)

    def block(prefix, R, base_lp, base_score, base_r, base_u, base_l):
        nonlocal mass
        # free bins: a = x_{n-2}, b = x_{n-1}, c = x_n = R - a - b  (1-based)
        ia = n - 3
        lo_a, hi_a = _cond_range(R, p[ia] / tails[ia], K)
        a = np.arange(lo_a, hi_a + 1)
        pb = p[ia + 1] / tails[ia + 1]
        lo_b, hi_b = R, 0
        for av in (lo_a, hi_a, int(R * p[ia] / tails[ia])):
            lb, hb = _cond_range(R - min(av, R), pb, K)
            lo_b, hi_b = min(lo_b, lb), max(hi_b, hb)
        b = np.arange(lo_b, max(lo_b, hi_b) + 1)
        A, B = np.meshgrid(a, b, indexing="ij")
        C = R - A - B
        ok = C >= 0
        A, B, C = A[ok], B[ok], C[ok]
        if A.size == 0:
            return
        lp = (base_lp - lg[A] - lg[B] - lg[C]
              + A * logp[ia] + B * logp[ia + 1] + C * logp[ia + 2])
        w = np.exp(lp)
        mass += w.sum()
        score = base_score + W[ia] * A + W[ia + 1] * B + W[ia + 2] * C
        last = prefix[-1]
        D = (A - last, B - A, C - B)
        Sr = base_r + sum(np.abs(L * D[k] - N * g[ia - 1 + k]) for k in range(3))
        Su = base_u + sum(np.abs(D[k]) for k in range(3))
        Sl = base_l + sum(sg[ia - 1 + k] * D[k] for k in range(3))
        ws = w * score
        for name, vals in (("d_r", Sr), ("d_u", Su), ("Lambda", Sl)):
            idx = vals + offs[name][0]
            size = offs[name][1]
            acc[name][0] += np.bincount(idx, weights=w, minlength=size)
            acc[name][1] += np.bincount(idx, weights=ws, minlength=size)

    def recurse(prefix, R, lp, score, sr, su, sl):
        i = len(prefix)               # 0-based index of the next bin
        if i == n - 3:
            block(prefix, R, lp, score, sr, su, sl)
            return
        lo, hi = _cond_range(R, p[i] / tails[i], K)
        for x in range(lo, hi + 1):
            d = x - prefix[-1]
            recurse(prefix + [x], R - x,
                    lp - lg[x] + x * logp[i], score + W[i] * x,
                    sr + abs(L * d - N * g[i - 1]), su + abs(d), sl + sg[i - 1] * d)

    recurse([x1], N - x1, const - lg[x1] + x1 * logp[0], W[0] * x1, 0, 0, 0)
    return acc, mass


def float_pass(N, q, K, pool):
    L = sum(q)
    lo, hi = _cond_range(N, q[0] / L, K)
    tasks = [(N, q, K, x1) for x1 in range(lo, hi + 1)]
    offs = statistic_offsets(N, q)
    tot = {s: [np.zeros(offs[s][1]), np.zeros(offs[s][1])] for s in STATS}
    mass = 0.0
    for acc, m in pool.imap_unordered(_worker, tasks):
        mass += m
        for s in STATS:
            tot[s][0] += acc[s][0]
            tot[s][1] += acc[s][1]
    return tot, mass


def summarize_float(tot, q, alpha, N):
    """Per statistic: cut, size, margin, slope (integer cut) and slope (randomized)."""
    _, factor, _ = comb_direction(q)
    factor = float(factor)
    offs = statistic_offsets(N, q)
    out = {}
    for s in STATS:
        P, M = tot[s]
        surv = np.cumsum(P[::-1])[::-1]            # surv[v] = P(T >= v)
        above = np.append(surv[1:], 0.0)            # above[v] = P(T > v)
        c = int(np.argmax(above <= alpha))
        size = above[c]
        margin = min(abs(above[c] - alpha), abs(above[c - 1] - alpha) if c > 0 else 1)
        slope_int = factor * M[c + 1:].sum()
        gamma = (alpha - size) / P[c]
        slope_rand = slope_int + gamma * factor * M[c]
        out[s] = dict(cut=c - offs[s][0], size=size, margin=margin,
                      slope=slope_int, slope_rand=slope_rand, gamma=gamma)
    return out


# ---------------------------------------------------------------- exact integer pass

def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def integer_pass(N, q, alpha):
    """Full enumeration of Mult(N, q/L) in exact integers (weights total L^N)."""
    n, L = len(q), sum(q)
    W, factor, _ = comb_direction(q)
    g = [q[i + 1] - q[i] for i in range(n - 1)]
    sg = [sgn(v) for v in g]
    fact = [1]
    for k in range(1, N + 1):
        fact.append(fact[-1] * k)
    powers = [[qi ** k for k in range(N + 1)] for qi in q]
    acc = {s: {} for s in STATS}
    for x in compositions(N, n):
        w = fact[N]
        for k, xk in enumerate(x):
            w //= fact[xk]
        for k, xk in enumerate(x):
            w *= powers[k][xk]
        sc = w * sum(Wk * xk for Wk, xk in zip(W, x))
        D = [x[i + 1] - x[i] for i in range(n - 1)]
        vals = {"d_r": sum(abs(L * D[i] - N * g[i]) for i in range(n - 1)),
                "d_u": sum(abs(d) for d in D),
                "Lambda": sum(sg[i] * D[i] for i in range(n - 1))}
        for s in STATS:
            e = acc[s].setdefault(vals[s], [0, 0])
            e[0] += w
            e[1] += sc
    total = L ** N
    out = {}
    for s in STATS:
        values = sorted(acc[s])
        cut = None
        # smallest v with P(T > v) <= alpha
        above = total
        for v in values:
            above -= acc[s][v][0]
            if Fraction(above, total) <= alpha:
                cut = v
                break
        size = Fraction(above, total)
        slope = factor * Fraction(sum(acc[s][v][1] for v in values if v > cut), total)
        gamma = (alpha - size) / Fraction(acc[s][cut][0], total)
        slope_rand = slope + gamma * factor * Fraction(acc[s][cut][1], total)
        out[s] = dict(cut=cut, size=size, slope=slope, slope_rand=slope_rand)
    return out


# ---------------------------------------------------------------- helpers

def loglog_slope(Ns, ys):
    xs = np.log(np.array(Ns, float))
    ys = np.array(ys, float)
    if np.any(ys <= 0):
        return float("nan")
    return float(np.polyfit(xs, np.log(ys), 1)[0])


def signed_fit(Ns, ys):
    """Log-log slope of |y|, or NaN if y changes sign over the grid."""
    if not all(np.sign(v) == np.sign(ys[0]) for v in ys):
        return float("nan")
    return loglog_slope(Ns, [abs(v) for v in ys])


def provenance(args, out):
    out("experiment_first_order_rate.py")
    out("command: python3 " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]))
    out(f"python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}")
    out("no seed: nothing is random.  Every row: enumeration of the multinomial in")
    out("float64; at N in " + str(args.int_N) + " also full enumeration in exact integers.")
    out(f"alpha = {args.alpha}; box = conditional mean +- {args.K} sd (+3) per bin")
    out()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--N", type=int, nargs="+", default=[20, 40, 80, 160, 320, 640])
    parser.add_argument("--int_N", type=int, nargs="*", default=[20, 40],
                        help="N at which the exact-integer cross-check is run")
    parser.add_argument("--gap_N", type=int, nargs="*", default=[20, 40, 80],
                        help="N at which the exact two-sign power gap of d_r is computed")
    parser.add_argument("--gap_small", type=str, default="1/50",
                        help="small target l1 amplitude for the power gap")
    parser.add_argument("--distance", type=str, default="1/5",
                        help="Table I target l1 distance")
    parser.add_argument("--alpha", type=str, default="1/20")
    parser.add_argument("--K", type=float, default=10.0)
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    parser.add_argument("--nulls", nargs="*", default=None,
                        help="subset of null names (default all)")
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results",
        "experiment_first_order_rate.txt"))
    args = parser.parse_args()
    alpha = Fraction(args.alpha)
    nulls = [x for x in NULLS if args.nulls is None or x[0] in args.nulls]
    lines = []

    def out(text=""):
        lines.append(text)
        print(text, flush=True)

    t0 = time.time()
    provenance(args, out)
    out("Slopes beta'(0) = Cov_H0(1{T>c}, sum_j delta_j x_j / p_j) along the UNIT-l1")
    out("multiplicative period-2 comb ('+' raises odd bins), each divided by sqrt(N).")
    out("int = integer cut c (size <= alpha); rnd = randomized cut (size = alpha exactly).")
    out("ratio = slope(d_r) / slope(Lambda).  'mass out' = multinomial mass outside the box")
    out("(1 - enumerated mass, clipped at 0; float64 rounding is ~1e-14).")
    out()

    summary = {}
    with Pool(args.jobs) as pool:
        for name, q in nulls:
            W, factor, delta = comb_direction(q)
            L = sum(q)
            g = [q[i + 1] - q[i] for i in range(len(q) - 1)]
            out(f"{name}   q = {':'.join(map(str, q))}   L = {L}")
            out("  delta = (" + ", ".join(f"{float(d):+.4f}" for d in delta) + ")"
                f"   Lambda = sum sgn(g_i) D_i, sgn(g) = {[sgn(v) for v in g]}")
            out(f"  {'N':>4} | {'stat':>6} {'cut':>6} {'size':>8} | {'int/sqrtN':>10} "
                f"{'rnd/sqrtN':>10} | {'mass out':>8}")
            rows = {}
            for N in args.N:
                t1 = time.time()
                tot, mass = float_pass(N, q, args.K, pool)
                res = summarize_float(tot, q, float(alpha), N)
                rows[N] = res
                for s in STATS:
                    r = res[s]
                    flag = "  (cut margin < 1e-9!)" if r["margin"] < 1e-9 else ""
                    out(f"  {N:>4} | {s:>6} {r['cut']:>6} {r['size']:>8.6f} | "
                        f"{r['slope'] / sqrt(N):>+10.5f} {r['slope_rand'] / sqrt(N):>+10.5f} | "
                        f"{max(1 - mass, 0.0):>8.1e}{flag}")
                ri = res["d_r"]["slope"] / res["Lambda"]["slope"]
                rr = res["d_r"]["slope_rand"] / res["Lambda"]["slope_rand"]
                ui = res["d_u"]["slope"] / res["Lambda"]["slope"]
                out(f"  {N:>4} | ratio d_r/Lambda  int {ri:+.5f}  rnd {rr:+.5f}   "
                    f"(d_u/Lambda int {ui:+.4f})   [{time.time() - t1:.1f} s]")
                print(f"[{name} N={N}: {time.time() - t1:.1f} s]", file=sys.stderr, flush=True)
            summary[name] = rows

            # exact integer cross-check
            for N in [m for m in args.int_N if m in rows]:
                t1 = time.time()
                ex = integer_pass(N, q, alpha)
                worst = 0.0
                same_cut = True
                for s in STATS:
                    fl = rows[N][s]
                    same_cut &= (fl["cut"] == ex[s]["cut"])
                    for key in ("slope", "slope_rand"):
                        e = float(ex[s][key])
                        worst = max(worst, abs(fl[key] - e) / max(abs(e), 1e-300))
                    worst = max(worst, abs(fl["size"] - float(ex[s]["size"])) / float(ex[s]["size"]))
                out(f"  exact-integer check N={N}: cuts {'identical' if same_cut else 'DIFFER'}, "
                    f"max relative difference of size/slopes = {worst:.1e}; exact d_r slope = "
                    f"{float(ex['d_r']['slope']):+.10f}  [{time.time() - t1:.1f} s]")
            out()

        # fitted slopes
        out("Log-log slopes in N (least squares over the N grid; NaN if a sign changes)")
        out("  quantity:  slope/sqrt(N) of d_r, Lambda, d_u; ratio d_r/Lambda.  The claim")
        out("  predicts -0.5 for d_r/sqrt(N) and the ratio, 0 for Lambda/sqrt(N).")
        out(f"  {'null':>11} | {'cut':>4} | {'d_r':>7} {'Lambda':>7} {'d_u':>7} | {'ratio':>7} | "
            "local exponent of the ratio per doubling")
        Ns = args.N
        for name, _ in nulls:
            rows = summary[name]
            for kind, key in (("int", "slope"), ("rnd", "slope_rand")):
                f = {s: [rows[N][s][key] / sqrt(N) for N in Ns] for s in STATS}
                ratio = [rows[N]["d_r"][key] / rows[N]["Lambda"][key] for N in Ns]
                sl = {s: signed_fit(Ns, f[s]) for s in STATS}
                sr = signed_fit(Ns, ratio)
                local = []
                for a, b in zip(range(len(Ns) - 1), range(1, len(Ns))):
                    if ratio[a] * ratio[b] > 0:
                        local.append(f"{log(abs(ratio[b] / ratio[a])) / log(Ns[b] / Ns[a]):+.2f}")
                    else:
                        local.append("sign")
                out(f"  {name:>11} | {kind:>4} | {sl['d_r']:>+7.3f} {sl['Lambda']:>+7.3f} "
                    f"{sl['d_u']:>+7.3f} | {sr:>+7.3f} | {' '.join(local)}")
        out()

    # exact power gap of d_r
    out("Exact two-sign power gap of d_r (dr_exact.py DP, alternative-centred; no floats).")
    out("a = exact l1 amplitude of p +- a delta; gap = power(+) - power(-);")
    out("linear = 2 a beta'(0) with the integer cut; the gap's odd part is 2a beta'(0) + O(a^3).")
    out(f"  {'null':>11} {'N':>4} {'a':>9} | {'cut':>5} {'power +':>9} {'power -':>9} "
        f"{'gap':>10} | {'linear':>10} {'gap/linear':>10}")
    for name, q in nulls:
        for target in (Fraction(args.distance), Fraction(args.gap_small)):
            plus, minus, a = comb_pair(q, target)
            for N in args.gap_N:
                L0 = sum(q)
                null = calculate_exact_dr_distribution(N, q)
                cut, size = tail(null, L0 ** N, alpha)
                pw = []
                for q1 in (plus, minus):
                    alt = calculate_exact_dr_distribution(N, q1, q)
                    pw.append(exceedance(alt, sum(q1) ** N, cut))
                gap = pw[0] - pw[1]
                slope = summary[name][N]["d_r"]["slope"] if N in summary.get(name, {}) else None
                lin = 2 * float(a) * slope if slope is not None else float("nan")
                out(f"  {name:>11} {N:>4} {float(a):>9.6f} | {cut:>5} {float(pw[0]):>9.6f} "
                    f"{float(pw[1]):>9.6f} {float(gap):>+10.6f} | {lin:>+10.6f} "
                    f"{float(gap) / lin if lin == lin and lin != 0 else float('nan'):>10.3f}")
    out()
    print(f"total runtime {time.time() - t0:.1f} s", file=sys.stderr)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
