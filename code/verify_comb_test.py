"""Self-test for comb_test.py.  Exits non-zero on any failure.

    python3 verify_comb_test.py

Groups: (1) 40 exact critical values and sizes against results/experiment_exact_power.txt;
(2) S = n * DTV on a uniform null; (3) S = L * ||x - Np||_V on random cases;
(4) T(t) identities; (5) collapse-index values (0 on a uniform null, near 1 on the
comb null, ordered on the Table I nulls); (6) p-value bounds and monotonicity;
(7) probs -> integer weights round-trip; (8) min-p: realized size near alpha
under H0 and a strong comb rejected, i.e. the LOWER-tail rule; (9) gamma: size on an
independent batch near alpha, cut near the exact one, no NaN on a degenerate null;
(10) the command-line program end to end, both routes agreeing on the same data,
the L cap routing many-decimal probabilities to the gamma, and the counts file;
(11) irrational nulls given as full-precision probabilities (an earlier TypeError in
the gamma route), the float statistic, the residue count c and the cost routing;
(12) the calibrated min-p p-value; (13) user-facing names (none of the test names
the Letter does not use, listed in test 13) and the documented example output.
"""
import re, sys, pathlib, subprocess, tempfile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from comb_test import (exact_tail, integer_statistic, collapse_index, family, min_p_calibrate,
                      gamma_tail, float_statistic, residue_count, min_p)
from dr_exact import calculate_exact_dr_distribution
from du_exact import prepare_integer_weights

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results"
fails = 0


def fail(msg):
    global fails
    fails += 1
    print("  FAIL", msg)


# --- 1. exact critical values and sizes, from experiment_exact_power.txt -------
txt = (RES / "experiment_exact_power.txt").read_text().splitlines()
cur, checked = None, 0
for line in txt:
    m = re.match(r"^(\S[\w -]*?)\s+q0=([\d:]+)", line)
    if m:
        cur = tuple(int(v) for v in m.group(2).split(":"))
        continue
    m = re.match(r"\s*(\d+) \|\s+(\d+) ([\d.]+) ([\d.]+) \|", line)
    if m and cur:
        N, crit, size = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if N > 40:            # keep the battery fast
            continue
        _, got_cut, got_size = exact_tail(N, list(cur), 0, 0.05)
        checked += 1
        if not (got_cut == crit and abs(got_size - size) < 5e-6):
            fail(f"q={cur} N={N}: expected crit {crit} size {size:.6f}, got {got_cut} {got_size:.6f}")
if checked < 40:
    fail(f"only {checked} exact cases found in experiment_exact_power.txt")
print(f"1. exact crit/size vs experiment_exact_power.txt: {checked} cases checked")

# --- 2. uniform null: S must equal n * DTV -----------------------------------
rng = np.random.default_rng(0)
for n, N in ((5, 12), (4, 15), (6, 10)):
    q = [1] * n
    x = list(rng.multinomial(N, [1 / n] * n))
    dtv = sum(abs(x[i] - x[i - 1]) for i in range(1, n))
    if integer_statistic(x, q) != n * dtv:
        fail(f"uniform n={n}: S={integer_statistic(x, q)} but n*DTV={n * dtv}")
print("2. uniform null S = n*DTV: 3 cases checked")

# --- 3. S = L * ||x - Np||_V, the Letter's Eq. (S), on random cases -----------
rng = np.random.default_rng(7)
for _ in range(200):
    n = int(rng.integers(2, 8))
    q = [int(v) for v in rng.integers(1, 9, size=n)]
    N = int(rng.integers(2, 60))
    L = sum(q)
    x = list(rng.multinomial(N, np.array(q) / L))
    r = np.array(x, float) - N * np.array(q, float) / L
    if abs(integer_statistic(x, q) - L * np.abs(np.diff(r)).sum()) > 1e-6:
        fail(f"S != L*||r||_V for q={q} x={x}")
print("3. S = L * ||x - Np||_V, 200 random cases: checked")

# --- 4. family(): T(0) is R; on a uniform null T(1) is proportional to T(0) ----
p = np.array([0.1, 0.4, 0.2, 0.3]); N = 50
x = np.array([[5., 20., 10., 15.]])
if abs(family(x, p, N, 0.0)[0] - np.abs(np.diff(x - N * p)).sum()) > 1e-9:
    fail("T(0) != R")
pu = np.full(4, 0.25)
xu = np.array([[10., 15., 12., 13.]])
if abs(family(xu, pu, N, 1.0)[0] - family(xu, pu, N, 0.0)[0] / np.sqrt(N * 0.25 * 0.75)) > 1e-9:
    fail("uniform null: T(1) is not proportional to T(0)")
print("4. T(t) family identities: checked")

# --- 5. collapse index: values, not just range --------------------------------
rng = np.random.default_rng(3)
if collapse_index([1, 1, 1, 1], 200, 4000, rng) != 0.0:
    fail("uniform null collapse index is not 0")
v_comb = collapse_index([1, 2, 1, 2, 1], 500, 20000, rng)
if not v_comb > 0.9:
    fail(f"comb null at N=500 should be near 1, got {v_comb:.3f}")
v_ramp40 = collapse_index([1, 2, 3, 4, 5], 40, 20000, rng)
v_run40 = collapse_index([16, 8, 4, 2, 1, 1], 40, 20000, rng)
v_comb40 = collapse_index([1, 2, 1, 2, 1], 40, 20000, rng)
if not (0.0 < v_ramp40 < v_run40 < v_comb40 < 1.0):
    fail(f"Table I ordering ramp < run-length < comb violated: {v_ramp40:.3f} {v_run40:.3f} {v_comb40:.3f}")
for q in ([1, 1000], [1, 2]):               # degenerate: must not return NaN
    v = collapse_index(q, 1, 2000, rng)
    if not np.isfinite(v):
        fail(f"collapse index NaN for q={q}, N=1")
print("5. collapse index values and degenerate guards: checked")

# --- 6. p-value monotonicity and bounds --------------------------------------
q = [1, 2, 1, 2, 1]; N = 20
prev = 0.0
for s in range(0, 200, 10):
    pv, _, _ = exact_tail(N, q, s, 0.05)
    if not (0.0 <= pv <= 1.0):
        fail(f"p-value {pv} out of [0,1] at s={s}")
    if s and pv > prev + 1e-12:
        fail(f"p-value not non-increasing at s={s}")
    prev = pv
print("6. p-value bounds and monotonicity: checked")

# --- 7. prepare_integer_weights round-trip -----------------------------------
for probs in ([0.5, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4], [1 / 7, 2 / 7, 1 / 7, 2 / 7, 1 / 7]):
    q = prepare_integer_weights(probs)
    if not np.allclose(np.array(q, float) / sum(q), probs, atol=1e-9):
        fail(f"weights round-trip {probs} -> {q}")
print("7. probs -> integer weights round-trip: checked")

# --- 8. min-p: size under H0 and the direction of the decision ----------------
q = [1, 2, 1, 2, 1]; N = 200; L = sum(q); p = np.array(q, float) / L
sizes = []
for seed in (11, 12, 13):
    rng = np.random.default_rng(seed)
    pmin, cut = min_p_calibrate(q, N, 0.05, 5000, rng)
    h0 = rng.multinomial(N, p, size=4000).astype(float)
    sizes.append(float((pmin(h0) <= cut).mean()))
    strong = np.array([40., 40., 40., 40., 40.])          # flat: the comb null's worst enemy
    if not pmin(strong[None, :])[0] <= cut:
        fail(f"min-p does not reject a flat histogram on the comb null (seed {seed})")
    if not (pmin(h0) <= cut).mean() < (pmin(h0) > cut).mean():
        fail("min-p rejects more often than not under H0: the rule is inverted")
if not all(0.03 <= s <= 0.07 for s in sizes):
    fail(f"min-p realized size under H0 off nominal: {sizes}")
print(f"8. min-p lower-tail rule: sizes {', '.join(f'{s:.3f}' for s in sizes)} at alpha 0.05, strong comb rejected")

# --- 9. gamma: size, cut near the exact one, degenerate null ------------------
rng = np.random.default_rng(5)
_, exact_cut, _ = exact_tail(40, [1, 2, 1, 2, 1], 0, 0.05)
pv, cut, size = gamma_tail(40, [1, 2, 1, 2, 1], 0, 0.05, 20000, rng)
cut *= 7                                   # gamma_tail works on d_r; the exact cut is on S = L*d_r
if not (0.035 <= size <= 0.065):
    fail(f"gamma size on an independent batch {size:.3f}")
if not abs(cut - exact_cut) / exact_cut < 0.10:
    fail(f"gamma cut {cut:.1f} far from the exact {exact_cut}")
if not pv == 1.0:
    fail(f"gamma p-value at s=0 should be 1, got {pv}")
pv, cut, size = gamma_tail(1, [1, 1], 1, 0.05, 2000, rng)   # d_r is constant here
if not (np.isfinite(pv) and np.isfinite(cut) and np.isfinite(size)):
    fail(f"gamma returns NaN on a degenerate null: {pv} {cut} {size}")
print("9. gamma size, cut, and degenerate null: checked")

# --- 10. the program end to end ----------------------------------------------
def run(*args, timeout=120):
    r = subprocess.run([sys.executable, str(HERE / "comb_test.py"), *args],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr

rc, out = run("--counts", "20", "20", "20", "20", "20", "--weights", "1", "2", "1", "2", "1",
              "--minp", "--draws", "5000", timeout=600)   # N = 100: exact in seconds on an idle machine
if rc != 0 or "EXACT" not in out:
    fail(f"basic run failed:\n{out}")
if out.count("REJECT the null") != 2:
    fail(f"exact route and min-p should both reject the flat histogram on the comb null:\n{out}")
if "low:" in out or "high:" in out:
    fail("collapse index is still printed with a threshold label")

rc, out = run("--counts", "2", "1", "1", "--probs", "0.3333333", "0.3333333", "0.3333334",
              "--draws", "2000", timeout=60)
if rc != 0 or "GAMMA" not in out or "L = " not in out:
    fail(f"many-decimal probabilities should route to the gamma quickly:\n{out}")

rc, out = run("--counts", "1", "0", "--weights", "1", "1", "--gamma", "--draws", "2000")
if rc != 0 or "nan" in out.lower():
    fail(f"degenerate gamma run printed NaN or failed:\n{out}")

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    f.write("12 5 9 4 11\n")
rc, out = run("--counts-file", f.name, "--weights", "1", "2", "1", "2", "1")
if rc != 0 or "N = 41" not in out:
    fail(f"--counts-file run failed:\n{out}")

rc, out = run("--counts", "1", "2", "3", "--probs", "0.2", "0.3", "0.4")
if rc != 0 or "renormalized" not in out:
    fail(f"probabilities not summing to one should be flagged:\n{out}")
print("10. command-line program, both routes, L cap, counts file, probability sum: checked")

# --- 11. irrational nulls, float statistic, residue count, cost routing -------
# Full-precision probabilities once became integer weights with a ~50-digit L,
# and scipy's gamma raised a TypeError on them.  They must take the float route.
w = np.exp(-0.5 * np.arange(6)); w /= w.sum()
rc, out = run("--counts", "3", "2", "1", "2", "1", "4", "--probs", *[repr(float(v)) for v in w],
              "--draws", "2000", timeout=60)
if rc != 0 or "Traceback" in out or "GAMMA" not in out or "digit integer" not in out:
    fail(f"repr(float) probabilities (n=6) should run on the gamma route:\n{out}")
if len(out) > 4000:
    fail(f"repr(float) probabilities printed {len(out)} characters (a huge q?)")
edges = np.linspace(-1.0, 1.0, 65)
arc = np.diff(np.arcsin(edges / 1.02) / np.pi); arc /= arc.sum()      # arcsine null, n = 64
xa = np.random.default_rng(1).multinomial(500, arc)
for extra in ((), ("--max-exact-n", "64", "--max-exact-N", "1000")):  # 2nd: only L blocks the exact
    rc, out = run("--counts", *map(str, xa), "--probs", *["%.10f" % v for v in arc],
                  "--draws", "2000", *extra, timeout=60)
    if rc != 0 or "Traceback" in out or "GAMMA" not in out or "nan" in out.lower():
        fail(f"arcsine null printed with %.10f should run on the gamma route {extra}:\n{out}")
# the float statistic is S/L on rational nulls and the gamma p-value is scale-free
rng = np.random.default_rng(17)
for _ in range(50):
    n = int(rng.integers(2, 9)); q = [int(v) for v in rng.integers(1, 9, size=n)]
    N = int(rng.integers(2, 60)); xr = list(rng.multinomial(N, np.array(q) / sum(q)))
    if abs(float_statistic(xr, q) - integer_statistic(xr, q) / sum(q)) > 1e-9:
        fail(f"float d_r != S/L for q={q} x={xr}")
pa, _, _ = gamma_tail(40, [1, 2, 1, 2, 1], 80 / 7, 0.05, 20000, np.random.default_rng(9))
pb, _, _ = gamma_tail(40, [v / 7 for v in (1, 2, 1, 2, 1)], 80 / 7, 0.05, 20000, np.random.default_rng(9))
if abs(pa - pb) > 1e-9:
    fail(f"gamma p-value depends on how the null is scaled: {pa} vs {pb}")
# c from the sign patterns equals the measured residue count of the exact null
for q, N in (([1, 2, 1, 2, 1], 40), ([16, 8, 4, 2, 1, 1], 40), ([16, 8, 4, 2, 1, 1], 20),
             ([1, 2, 3, 4, 5], 20), ([1, 2, 3, 2, 1], 20), ([1, 1, 1, 1], 30), ([7, 3, 5, 2], 25)):
    L = sum(q)
    measured = len({v % L for v, k in calculate_exact_dr_distribution(N, q) if k})
    c = residue_count(N, q)
    if c != measured or c > min(L, 2 ** (len(q) - 1)):
        fail(f"residue count q={q} N={N}: computed {c}, measured {measured}")
rc, out = run("--counts", "8", "12", "8", "12", "0", "--weights", "1", "2", "1", "2", "1",
              "--max-exact-seconds", "0", "--draws", "2000")
if rc != 0 or "GAMMA" not in out or "estimated exact run time" not in out:
    fail(f"the cost cap should route to the gamma:\n{out}")
print("11. irrational nulls (repr and %.10f probabilities), float d_r, residue count c, cost cap: checked")

# --- 12. min-p: the calibrated p-value -----------------------------------------
q = [1, 2, 1, 2, 1]; N = 200; p = np.array(q, float) / sum(q)
rng = np.random.default_rng(21)
pmin, cut, null2 = min_p_calibrate(q, N, 0.05, 5000, rng, return_null=True)
h0 = rng.multinomial(N, p, size=4000).astype(float)
obs = pmin(h0)
srt = np.sort(null2)
pv = (1.0 + np.searchsorted(srt, obs, "right")) / (len(srt) + 1.0)
size = float((pv <= 0.05).mean())
if not (0.03 <= size <= 0.06) or pv.min() <= 0 or pv.max() > 1:
    fail(f"calibrated min-p p-value: size {size:.3f}, range {pv.min()}..{pv.max()}")
o, _, pflat = min_p([40, 40, 40, 40, 40], q, N, 0.05, 5000, np.random.default_rng(22))
if not pflat <= 0.05:
    fail(f"calibrated min-p p-value of a flat histogram on the comb null is {pflat}")
print(f"12. calibrated min-p p-value: size {size:.3f} at alpha 0.05, flat histogram rejected")

# --- 13. names and the documented example -------------------------------------
outs = [run("--help")[1]]
rc, out = run("--counts", "8", "12", "8", "12", "0", "--weights", "1", "2", "1", "2", "1", "--minp",
              "--draws", "5000")
outs.append(out)
outs.append(run("--counts", "8", "12", "8", "12", "0", "--probs", "0.1", "0.2", "0.1", "0.2", "0.4",
                "--gamma", "--draws", "2000")[1])
# Former test names that must never be printed.  Built by concatenation on purpose,
# so that a repository-wide grep for former names does not hit this guard.
FORBIDDEN = ["resid" + "ual comb test", "uncentered comb test", "resid" + "ual test"]
for o in outs:
    low = o.lower()
    if any(name in low for name in FORBIDDEN):
        fail(f"a non-Letter test name is printed:\n{o}")
if "comb test (CT)" not in out or "uncentered DTV" not in out or "NOT a p-value" not in out:
    fail(f"output lacks the Letter's names or the min-p label:\n{out}")
if ("reject when S > 165" not in out or "attained size 0.0495" not in out
        or "at N = 40: 0.96" not in out or "Pr[S >= 80] = 0.5594" not in out):
    fail(f"documented comb-null example (cut 165, size 0.0495, index 0.96) changed:\n{out}")
if "Pr[S >=" in outs[2] or "gamma fit, approximate" not in outs[2]:
    fail(f"gamma route must not label its p-value as an exact Pr[S >= s]:\n{outs[2]}")
print("13. names, p-value labels and the documented comb-null example: checked")

print("\nFAILURES:", fails)
sys.exit(1 if fails else 0)
