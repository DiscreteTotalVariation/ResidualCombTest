# Reproducing the Letter

Details behind `README.md`: notation, the full options of `comb_test.py`, every
experiment, environment, seeds, run times, and where each number of the Letter
comes from.

## Names and notation

The test is the *comb test* (CT) with null `r`: the discrete total
variation of the centered counts `e = x - N r`, `d_r(x) = ||x - N r||_V`, which
the Letter calls the *centered total variation* (TVC).  For the uniform null `u`
it is the published statistic `d_u(x) = ||x||_V`.  **Notation in this
repository:** the code and the result files write the null as `p` and the
centered counts as `r_i = x_i - N p_i`; `S = L*d_r` is the integer lattice value
of the statistic.  File names follow the Letter's notation (`dr_exact.py` for the
exact null of `d_r`, `du_exact.py` for that of `d_u`, `comb_test.py` for the
command-line test).  Result-file column headers: `CT` and `T(0)` are the CT,
i.e. `d_r`; `T(0.5)` and `T(1)` are the other members of the family of
Section IV-B; `du` and `DTV` are the uncentered DTV `d_u`, which is not a test of
the non-uniform null; `DTV2` is a two-sided rule on `d_u`, a repository-only
negative control.  In the code and result files the a-priori diagnostic that
predicts how much the uncentered DTV would lose of its phase-blindness is called
`linearity`; in the Letter it is called the *collapse index*.  Sums over
neighbouring bins run `i = 1..n-1` in the Letter (`|x_{i+1} - x_i|`); some code
comments index the same terms `i = 2..n` (`|x_i - x_{i-1}|`).

## comb_test.py in detail

Probabilities are converted to the smallest integer weights (`0.1 0.2 0.1 0.2
0.4` becomes `q = (1, 2, 1, 2, 4)`, `L = 10`), and the statistic is reported on
that integer lattice as `S = L*d_r`, with `d_r = S/L` (Eq. (3) of the Letter).

It reports the statistic `d_r` and its lattice value `S`, the critical value, the
p-value, a decision at `--alpha` (default 0.05), and the **collapse index** of your
null at your `N`: 0 means the uncentered DTV `d_u` would have lost nothing on it,
1 that it would have collapsed into a signed sum and detect only one comb sign. The index *orders*
the loss (Section III); there is no threshold.  The CT itself is centered and does
not collapse.  `--counts-file` reads the counts from a text file instead.

Calibration is chosen for you: the **exact** null of Section II when `N`, the
number of bins and the weight total `L` are within the recurrence's reach
(`--max-exact-N` 200, `--max-exact-n` 12, `--max-exact-L` 1000, and an
estimated run time within `--max-exact-seconds`, default 1800), and the
moment-matched **gamma** of Section IV-A beyond it. Probabilities whose lowest-terms
denominator exceeds `L = 1000` (typically four or more decimals) take the gamma
route; give the null as `--weights` for an exact null. `--gamma` forces the
gamma; `--minp` adds Tippett's combination over `T(0)`, `T(0.5)`, `T(1)`, which is
what Section IV-B recommends when you do not know whether the departure scales the
probabilities or shifts them.  The minimum of the three member p-values is not
itself a p-value: the program calibrates it by Monte Carlo on an independent
second batch of null draws, prints the **calibrated p-value**, and rejects when
that p-value is **at or below** `--alpha`.  Choose between the CT and min-p before
looking at the data.

**Irrational nulls.**  A null that is not a small-denominator rational (an arcsine
code density, a Gaussian stimulus) can be given to full precision with `--probs`;
it is then used in floating point and calibrated by the gamma of Section IV-A, at
any `n` and `N`.  For an exact null instead, round it to integer weights
`q_i ~ L r_i` (for example by largest-remainder rounding, so that the `q_i` sum to
`L`) and pass them as `--weights`.  This shifts the centering `N r_i` by less than
`N/L` per bin, and the exact null then costs about `n N^4 c` operations, with
`c <= min(L, 2^(n-1))` the number of residue classes of `S` mod `L` (Section II-A);
for a finely rounded null `c` is usually at its cap, so exactness is practical
only for small `n`.  Before a long exact run `comb_test.py` prints `c` and an
estimated run time, and it switches to the gamma when that estimate exceeds
`--max-exact-seconds`.

**The null must be known, not estimated** from the same data or from a short clean
capture -- see the calibration-capture section of `GLOSSARY.md`, where estimating
it from a capture the length of the test sample drives the size from 0.05 to 0.65.

`code/verify_comb_test.py` is its self-test, thirteen groups: 40 exact critical
values and sizes checked against `results/experiment_exact_power.txt`, the
uniform reduction and the statistic and family identities, collapse-index values,
p-value bounds, the probability-to-weight round trip, the min-p rule's realized
size under the null and its direction, the gamma's size and a degenerate null, the
program end to end on both routes, irrational nulls and the residue count `c`, the
calibrated min-p p-value, and the user-facing names and documented example
output. It runs as part of `pipeline.sh --check`.

## Layout

| path | contents |
|---|---|
| `code/comb_test.py` | the comb test on one histogram (see `README.md`); `verify_comb_test.py` is its self-test |
| `code/du_exact.py` | exact null of the uncentered DTV `d_u` under a non-uniform rational null (dynamic program); also `prepare_integer_weights` |
| `code/dr_exact.py` | **exact null of the CT, as `S = L*d_r` (Eqs. (3)-(4))**, under a non-uniform rational null; `--centre` gives exact power |
| `code/circular_du_exact.py` | exact null of the uncentered *circular* DTV (adds the wrap-around edge `|x_1 - x_n|`) under a non-uniform rational null; writes `data/exact_distributions_circular/`, parked and not used by the Letter (the circular DTV is future work) |
| `code/dr_demo.py` | the original exploratory Monte Carlo demo (seed 31, no arguments, no result file, not in the pipeline): Part 1 measures how far `d_u` locks into its signed sum on four nulls, Part 2 compares the CT `= T(0)` and `T(1)` against additive and multiplicative combs |
| `code/verify_dr_exact.py` | seven checks of `dr_exact.py` and of the claims about it: mass, uniform reduction, brute force, Monte Carlo, `S = L*d_r`, invariance of the residue count under `q -> lambda*q` (note: this one is an algebraic identity, not evidence about cost -- see `experiment_dp_complexity.py` for the substantive test), and Proposition 1 (variance, sign-locking bound) |
| `code/du_exact_bf.py`, `du_exact_bf2.py`, `du_mc.py`, `ct_bf.py`, `ct_bf2.py` | brute-force and Monte Carlo cross-checks |
| `code/generate_exact_*.sh` | drivers that fill `data/` (see `data.md`) |
| `code/experiment_*.py` | every experiment in the letter and the repository-only ones (see below) |
| `results/*.txt` | captured output of every script; **every number in the letter traces to a line here** |
| `paper/img/`, `paper/table*.tex` | figure PDFs and table bodies written by the pipeline (the manuscript itself is not part of this repository) |
| `figures/`, `tables/` | one script per figure / table of the letter; the pipeline regenerates all of them.  `figures/fig_main.py` draws Fig. 1; `figures/fig_appendix.py` draws `paper/img/fig_appendix.pdf`, the upper tail of the CT's exact null against its moment-matched gamma (run-length null, `N = 100`, recomputed from `dr_exact.py` in seconds), which is not in the Letter |
| `data/` | precomputed exact distributions; gitignored, regenerated on demand (see `data.md`) |

## Experiments

| script | what it shows | result file |
|---|---|---|
| `experiment_exact_power.py` | Table I: exact size and power, both comb signs at identical l1, the CT against the uncentered `d_u` (one-sided, as in the Letter; the two-sided `d_u` is a repository-only negative control) | `experiment_exact_power.txt` |
| `experiment_degeneracy.py` | Fig. 1a: the uncentered `d_u` loses phase-blindness as the null steepens; the CT's two comb-sign powers stay within 0.02 of each other at `N = 1000` (within 0.04 at `N = 50, 200`) | `experiment_degeneracy.txt` |
| `experiment_rho_diagnostic.py` | the collapse index (`linearity`) vs the phase-power ratio; image nulls on both sides | `experiment_rho_diagnostic*.txt` |
| `experiment_exact_index.py` | the collapse index of Table I's five nulls at `N = 40`, EXACT by enumerating the at-most-four-bin marginals every moment needs; no seed | `experiment_exact_index.txt` |
| `experiment_rho_sweep.py` | 42 nulls spanning the index range: the index ORDERS the loss of `d_u` (no threshold); the CT's worse-sign power stays between 0.78 and 0.97 over all 42 | `experiment_rho_sweep.txt` |
| `experiment_bias.py` | exact size and power over whole lattices of alternatives: how often each test's power falls BELOW its size (Good and Smith 1984) | `experiment_bias.txt` |
| `experiment_runlength_rng.py --skip_b --N 200 1000 --exact_N 100 200 --bits 8000000` | Part A at 8e6 bits, a shorter side run.  The Letter quotes the DEFAULT 4e7-bit run (0.169, 0.603); this file gives 0.171 and 0.606.  The exact quantities (cut 1560, size 0.0485) are identical in both | `experiment_runlength_rng_partA_exact.txt` |
| `experiment_runlength_rng.py --skip_a --bits 4000000 --window_N 1000 2500 --random_phase --eps 0 0.05 0.07 0.10` | the random-lane comparison on 5000-bit windows (a third window width, checked for the "in every window" statement) | `experiment_runlength_rng_randomphase_5000bits.txt` |
| `experiment_runlength_rng.py --skip_a --random_phase` | the same comparison with each window starting on a random lane, as a real capture would.  The lane phase matters: held fixed it aligns the longest-run test's blocks with the defect | `experiment_runlength_rng_randomphase.txt` |
| `experiment_rho_seeds.py` | repeats the collapse-index sweep at five seeds, so the Letter can quote a RANGE for the rank correlation instead of a two-decimal point estimate that does not reproduce | `experiment_rho_seeds.txt` |
| `experiment_timing.py` | wall-clock cost of the exact DP and the reach of exactness.  One of two results files that will NOT reproduce byte for byte (the other is `experiment_dp_complexity.txt`, whose timing columns move), since timings are machine-dependent | `exact_reach_timing.txt` |
| `experiment_bias_exhaustive.py` | the same question with NO randomness: complete families of rational nulls against every period-2 comb on a fixed amplitude grid, 247,608 exact cases.  This is the file the Letter's bias sentence quotes, because a worst case over randomly drawn nulls moves with the seed | `experiment_bias_exhaustive.txt` |
| `verify_proposition1.py` | Proposition 1 by exhaustive enumeration in exact rationals: the Hoeffding bound on a sign flip with its true slack, the union bound on `Pr[E]`, `d_u = Lambda` (Eq. (5)) on the locking event `E`, and the telescoping to `+-(x_n - x_1)` for monotone nulls; it also checks the identity `Var[D_i] = N(p_i + p_{i+1} - g_i^2)`, which is not part of the Letter's statement | `verify_proposition1.txt` |
| `experiment_stress.py` | large-scale randomized re-tests of the Letter's near-universal claims: exactness against brute force, the alternative-centred path, degenerate inputs, the residue bound, p-value validity, **the recurrence AS PRINTED in the Letter (Eq. (4))**, gamma at other levels, the index across `N`, the bias mechanism | `experiment_stress_*.txt` |
| `experiment_robustness.py` | how the test behaves outside the Letter's exact setting: discreteness penalty, the cost of a two-sided centered test, gamma against the EXACT critical value, a DRIFTING null (repository only), power against `N`, localized combs | `experiment_robustness_*.txt` |
| `experiment_design.py` | the design choices a referee can see: run-length pooling depth, the AC oracle at other periods, other code counts, the zero-weight boundary | `experiment_design_*.txt` |
| `experiment_smooth.py` | the competitor a referee is likeliest to propose: Neyman's smooth test on the top `k` frequencies, which needs no knowledge of the period | `experiment_smooth.txt` |
| `experiment_dp_complexity.py` | support and running time of the exact DP vs `N` and `L` | `experiment_dp_complexity.txt` |
| `experiment_null_approximation.py` | gamma vs beta fit to the exact null | `experiment_null_approximation.txt` |
| `experiment_gamma_large_n.py` | gamma false-positive rate to `N = 5e4` for `T(0)`, `T(0.5)`, `T(1)` | `experiment_gamma_large_n.txt` |
| `experiment_selection_rule.py` | the min-p combination and its worst-case regret | `experiment_selection_rule*.txt` |
| `check_minp.py` | why the min-p minimum is MC-calibrated (Sidak/Bonferroni are conservative; `--part sidak`) and that it stays phase-blind (`--part phase`) | `check_minp_sidak.txt`, `check_minp_phase.txt` |
| `experiment_shape_power.py` | l1-matched power over departure shapes, order-aware and ordinal baselines | `experiment_shape_power.txt` |
| `experiment_runlength_rng.py` | run-length histograms of a two-lane bit source vs NIST tests on the same bits | `experiment_runlength_rng*.txt` |
| `experiment_real_images.py` | code histograms of real images without a rank transform; `--images` for per-image | `experiment_real_images*.txt` |
| `experiment_sine_histogram_adc.py` | IEEE 1241 sine-wave histogram test, arcsine null, SAR-mismatch comb | `experiment_sine_histogram_adc*.txt` |
| `experiment_double_jpeg.py` | double-quantization comb in DCT histograms (repository only) | `experiment_double_jpeg.txt` |
| `experiment_local_sensitivity.py` | directions of maximum and minimum local sensitivity (repository only; background for Sec. V-B) | `experiment_local_sensitivity.txt` |
| `experiment_variants.py` | defect geometry, sparsity, null misspecification (repository only) | `experiment_variants.txt` |
| `experiment_period_competitors.py` | the CT against the alternating contrast AC, `X^2`, pooled `X^2` and a min-p of AC and `X^2`, at period 2 and period 4: Block 1 the five image nulls (`n = 64`, `N = 2000`, `eps = +-0.06`), Block 2 the four synthetic nulls of `experiment_design.py --part acperiod` (periods 2, 3, 4, 6).  Uses the parent scripts' seeds and draw order, so its period-2 image rows and every acperiod row reproduce those files draw for draw, and it prints the differences; the `_seedcheck` run repeats it on other seeds (`--seed_images 1019 --seed_synthetic 1016`), where the reproduction becomes a check within MC error | `experiment_period_competitors.txt`, `experiment_period_competitors_seedcheck.txt` |
| `experiment_bitsource_lanetests.py` | lane-aware standard tests on the two-lane bit source of Section V-C, on the same generator, windows (400, 2000, 10000 bits) and random starting lane: the CT on the run-length histogram against Knuth's serial test on non-overlapping bit pairs and the exact flip-parity test; also re-checks the CT's earlier random-lane powers.  Run with no arguments, seed 33 | `experiment_bitsource_lanetests.txt` |
| `experiment_sine_fitted_null.py` | the IEEE 1241 sine-wave row (`n = 64`, `N = 10^4`, `eps = +-0.05`) with the arcsine null FITTED from each record by the standard's end-code estimator instead of the true amplitude and offset: size and power of the CT and `X^2` with the true-null cut, an oracle cut and a per-record parametric bootstrap.  Run with no arguments, seed 33 | `experiment_sine_fitted_null.txt` |
| `experiment_first_order_rate.py` | Section III's first-order claim, with no Monte Carlo: the slope at zero amplitude of the rejection rate of `d_r`, `d_u` and `Lambda` along the comb direction, by enumerating the multinomial (in exact integers as well at `N = 20, 40`), and the exact two-sign power gap of the CT from `dr_exact.py` against its linear prediction.  `python3 experiment_first_order_rate.py --jobs 16`; deterministic | `experiment_first_order_rate.txt` |

Run from `code/` with `python3 <script>.py`; seeds are fixed.  Python 3 with
`numpy`, `scipy`, `matplotlib`, and `scikit-image`.

## Environment

`requirements.txt` gives version floors; `requirements-lock.txt` records the exact
versions that produced `results/`.  Byte-identical reproduction of the Monte Carlo
files needs the same numpy, because the generator's stream is version-dependent.
The exact results do not depend on any of this.

The image experiments read their nulls from scikit-image's bundled sample images
(`camera`, `coins`, `moon`, `page`, `cell`): the null IS the shipped image's own
code histogram, so pinning that version matters for those numbers.

## Seeds

Every Monte Carlo seed of the original experiments is `13 + i` with `i` the
script's alphabetical index; see `code/SEEDS.md` for the table. One documented
root, one independent stream per experiment. Most take a `--seed` argument;
`verify_dr_exact.py` and `dr_demo.py` hardcode theirs, and
`experiment_rho_seeds.py` takes `--seeds`.  The later scripts sit outside that
numbering so that no existing seed moves: `experiment_period_competitors.py`
reuses its parent scripts' seeds 19 and 16 by design (draw-for-draw
reproduction), `experiment_bitsource_lanetests.py` and
`experiment_sine_fitted_null.py` both take 33 (independent experiments that share
no draws), and `experiment_first_order_rate.py`
draws nothing.

## Scripts that need an argument

`experiment_stress.py`, `experiment_robustness.py`, `experiment_design.py` and
`check_minp.py` each run one part at a time and exit 2 without one, e.g.

    python3 experiment_stress.py --part exact --cases 150

`pipeline.sh --stress` passes the right part and the right case count for every
one of them; run bare they default to fewer cases than the shipped results used.

## Run modes and run times

`--stress` (about 75 min: about 50 min for the randomized and exhaustive checks,
plus about 13 min for `experiment_bitsource_lanetests.py` and 12 min for
`experiment_first_order_rate.py` on 16 cores) runs the large randomized and
exhaustive checks, and is the ONLY mode that runs `verify_proposition1.py`,
`experiment_bias_exhaustive.py`, `experiment_timing.py`, `experiment_rho_seeds.py`,
`experiment_smooth.py`, `experiment_robustness.py`, `experiment_design.py`,
`experiment_bitsource_lanetests.py` and `experiment_first_order_rate.py`.  It
covers the bias survey, phase-blindness, robustness, design choices, the
lane-aware bit-source tests and the first-order rate; several numbers the Letter
cites as "(repository)" come only from that mode.  Run it after `--from-scratch`
(or on the shipped `results/`): `experiment_bitsource_lanetests.py` compares
against the random-lane run-length files that mode writes, and
`experiment_period_competitors.py` (run by `--from-scratch`) against the image
and `acperiod` files.  `--tables-only` regenerates the two generated tables.


```
bash pipeline.sh --check          # exact-DP verification + determinism check (seconds)
bash pipeline.sh --plots-only     # regenerate paper/img/*.pdf and paper/table*.tex from results/
bash pipeline.sh --from-scratch   # rerun every experiment (about an hour), then plots
./pipeline_docker.sh --plots-only # the same inside Docker (the image installs the URW fonts so the figures match)
```

Runtimes worth knowing: `experiment_runlength_rng.py` about 10 min; the exact
dynamic program at N = 200 about 90 s; `experiment_sine_fitted_null.py` about
3 min; `experiment_period_competitors.py` about 12 s per run (multithreaded numpy);
`experiment_bitsource_lanetests.py` about 13 min; `experiment_first_order_rate.py`
about 12 min on 16 cores; everything else seconds to a few minutes.
See `data.md` for file formats.

## What reproduces byte for byte

With the versions in `requirements-lock.txt`,
every results file reproduces byte for byte, Monte Carlo ones included, except for
wall-clock timings: the seconds columns and fitted time slopes of
`experiment_dp_complexity.txt`, the run times in `exact_reach_timing.txt`, and the
run-time stamps of `experiment_sine_fitted_null.txt` and
`experiment_first_order_rate.txt` (whose last digit of the float-noise "mass out"
column can also move with `--jobs`).  The Letter's cost ratios (Sec. II-A,
2.5 / 2.0 / 1.3, from `exact_reach_timing.txt`) are timing ratios: they hold on an
otherwise idle machine and drift by a few tenths under load, while the support
and residue-class columns beside them are exact.  The "max relative difference"
of the exact-integer checks in `experiment_first_order_rate.txt` (about 1e-14) is
float64 rounding of the same kind.

## Where each (repository) number comes from

The Letter marks with "(repository)" the claims whose evidence is not printed in it.
Each is below, together with the derived ranges and the numbers that only `--stress` produces; FS = `--from-scratch`,
ST = `--stress`.  Line numbers refer to the shipped `results/`.

| Letter (section): claim | file, place | how the quoted value is read | mode |
|---|---|---|---|
| III: the CT's two-sign power gap has a derivative at most of order N^{-1/2} times that of Lambda | `experiment_first_order_rate.txt` l.104-113 | log-log slope of the ratio d_r/Lambda: ramp -0.59 / -0.45, run-length -0.49 / -0.50 (int / rnd cut); comb null: sign changes, ratio at most 0.11 in magnitude | ST |
| III: two-sided d_u leaves the signs unequal and loses power on the flat null | `experiment_degeneracy.txt` l.25-26 (wDTV2 0.484 < wDTV 0.586 at beta = 0); `experiment_real_images_per_image.txt`, column DTV2 per sign | worse-sign two-sided vs one-sided power; per-sign values in the image file | FS |
| III: Spearman -0.90 to -0.84 over five seeds | `experiment_rho_seeds.txt` l.4-11 | `scipy.stats.spearmanr` of index vs d_u worse-sign power, one value per seed; range printed on the last line | ST |
| II-A: cost ratio 2.5 / 2.0 / 1.3, c from 8 to 2 | `exact_reach_timing.txt` l.5-8 | columns ratio and c (ratio is wall clock, see above) | ST |
| IV-A: gamma cut at alpha 0.01 and 0.001 | `experiment_robustness_gammaexact.txt` | exact cut vs gamma cut, exact size at the gamma cut | ST |
| V-A: bias survey over 247 608 cases | `experiment_bias_exhaustive.txt` l.61, 64 | counts printed directly | ST |
| V-B: smooth test on 2-4 frequencies trails the CT on four of five nulls | `experiment_smooth.txt`, comb rows (l.8, 13, 18, 23, 28) | CT > smooth2..4 on all but the geometric null | ST |
| V-B: CvM and AD 0.06-0.34 on combs, 0.58-0.88 on tilts | `experiment_shape_power.txt`, comb and "tilt (linear)" rows | columns CvM and AD; on combs the worse of the two signs per null, then min and max over nulls | FS |
| V-B: CT / pooled X^2 from 1.63 to 2.50 on ten images | `experiment_stress_images.txt` l.14 | printed directly | ST |
| V-B: period-4 code widths: AC and min-p(AC, X^2) 0.14 or less, CT 0.14-0.56, ahead of pooled X^2 on two | `experiment_period_competitors.txt`, Block 1, rows with per = 4 | columns CT, AC, X2p, mp(AC,X2); CT > X2p on moon and cell | FS |
| V-B: fitted arcsine null: size 0.025, power 0.65, X^2 0.44; bootstrap size 0.048, power 0.74 | `experiment_sine_fitted_null.txt` l.22, 24, 29 | rows "CT fit/trueMC", "X2 fit/trueMC", "CT fit/boot" | FS |
| V-C: frequency, runs, spectral, serial(m=2) within 0.024-0.076, lane fixed or drawn | `experiment_runlength_rng.txt` Part B and `experiment_runlength_rng_randomphase*.txt` | min and max of columns mono, runs, dft, ser2 over all windows and eps | FS |
| V-C: lane-aware tests (pair serial, even/odd parity) 0.76 and 0.99 against the CT's 0.09 | `experiment_bitsource_lanetests.txt` l.34 (eps = 0.05, 2000-bit windows, lane drawn) | columns pair, parity, CT | ST |
| VI: a calibration capture of the sample's length inflates the size of the CT and X^2; 125 times longer removes it for the CT | `experiment_stress_misspec.txt` l.6 (M0/N = 1: CT 0.650, X^2 0.872), l.9 (125: CT 0.053) | printed directly | ST |

All the other numbers in the Letter, and both tables and Fig. 1, come from FS
(tables and figure via `--plots-only`).

## Reproducing (manual)

```
cd code
python3 verify_dr_exact.py            # must report ALL CHECKS PASSED (seven groups)
python3 experiment_exact_power.py      # deterministic; diff against results/
```
