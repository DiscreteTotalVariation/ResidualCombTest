# Terms used in the Letter

> **Names and notation.** The Letter calls the test the *comb test* (CT) with null
> **r** and its statistic `d_r = ‖x − Nr‖_V`, the *centered total variation* (TVC);
> the uncentered DTV `d_u = ‖x‖_V` is not a test of a non-uniform null, only what
> the Letter's Section III analyses. This glossary, the code and the result files
> write the null as `p` and the centered counts as `r = x − Np`. Sums over neighbouring bins run `i = 1..n−1`, with the
> Letter's forward differences: expected slope `g_i = p_{i+1} − p_i`, observed gap
> `D_i = x_{i+1} − x_i`, so `E[D_i] = N g_i`.
>
> **Result-file column headers:** `CT`, `T(0)` = `d_r`; `T(0.5)`, `T(1)` = the
> other members of the T(t) family; `du`, `DTV` = `d_u`; `DTV2` = a two-sided rule on
> `d_u`, a repository-only negative control (the Letter's Table I has no
> two-sided column).

The Letter is four pages and defines each term once, in passing. This file is the
longer version: every term a reader has had to stop and decode, with the reason it
matters. Numbers quoted here are the Letter's own and trace to `results/`.

## Size, level, power

**Size** is the probability that a test rejects **when the null is true** — its
actual false-alarm rate. A test of size 0.047 wrongly cries "defect" on 4.7 % of
clean data.

**Level α** is the size you *ask* for, usually 0.05. With a continuous statistic
you hit it exactly. The statistic `S` is integer-valued, so the achievable
rejection probabilities come in jumps and you take the largest one not exceeding
α — which is why attained sizes read 0.042, 0.047, 0.050 and never exactly 0.05.
A test is **conservative** when size < α (safe, but power is lost) and
**anti-conservative** when size > α (invalid: any power advantage it shows is
partly bought by over-rejecting).

**Power** is the probability of rejecting when the null is **false**.

Against the machine-learning vocabulary:

|                    | null true                    | null false               |
|--------------------|------------------------------|--------------------------|
| **reject**         | size = FPR = Type I error    | power = TPR = sensitivity|
| **do not reject**  | TNR = specificity = 1 − size | FNR = Type II error      |

So size is the **false** positive rate, not the true positive rate, and
specificity is its complement. **"Power below size"** — the dagger in Table I —
means TPR < FPR: the test fires more readily on clean data than on defective
data. That is worse than useless, and in Table I it marks the uncentered `d_u`
on the run-length and comb nulls.

**"Holds size"** means the size stays at or near nominal when the setup is
perturbed. One perturbation, studied in the repository only
(`experiment_robustness.py --part drift`), is null drift: the critical value is
calibrated on one null and the true null then moves by ℓ1 distance 0.04–0.05. The
CT's size rises to 0.055–0.116, chi-square's to 0.36–0.47. Note that 0.116 is
more than twice nominal — the CT degrades gracefully, it does not stay valid.

## The comb and its two signs

A **comb** is a period-2 departure, `δ_i = (−1)^i δ`: every second bin raised, the
rest lowered. It is what a converter with a mismatched least-significant capacitor
produces in its code histogram.

For a given amplitude there are **two** combs, one per parity of raised bins (on a
non-uniform null they are not shifted copies of each other: `comb_pair` in
`experiment_exact_power.py` uses the same δ with opposite sign). The Letter calls
them the **comb signs**:

- **sign +** is `p + δ`, which raises bins **1, 3, 5, …**
- **sign −** is `p − δ`, which raises bins **2, 4, …**

(Verified against `experiment_exact_power.py`; the code builds the pattern
0-based, the Letter indexes bins 1-based, so read across that boundary with care.)

They are at equal ℓ1 distance from the null and differ in which parity is raised.

**Worse sign / worse-sign power** — every test has two powers, one per sign. The
worse-sign power is the smaller of the two: the worst case over the comb's phase.
The Letter reports it because a practitioner does not know which phase their
defect has, so quoting the favorable sign would be cherry-picking. On the comb
null the uncentered `d_u`'s two powers are 0.332 and 0.002, so the choice decides the
entire story.

**Phase-blindness** is insensitivity to which sign is present — the property that
makes a comb test usable when you cannot know the phase in advance.

## "Collapses to a fixed signed sum"

The phrase in Section I, unpacked.

The DTV is `Σ_{i=1}^{n−1} |x_{i+1} − x_i|` (Eq. (1)). Write `D_i = x_{i+1} − x_i`
for the observed gap between neighboring bins. The absolute value is a **fold**: it sends +3 and −3 to
the same place, and that fold is exactly what makes the test indifferent to *which*
bins are raised.

Now put a steep null underneath. The expected gap is `N·g_i`, with
`g_i = p_{i+1} − p_i`, its sampling noise about `sqrt(N(p_i + p_{i+1}))`. Once the
expected gap exceeds that noise, `D_i` essentially never crosses zero — a bin that
should be higher than its neighbour always is. Proposition 1 bounds the chance of a
crossing by `exp(−N g_i² / 2)`.

If `D_i` never changes sign then

    |D_i| = sgn(g_i) · D_i

and the fold has vanished: the absolute value has become **multiplication by a
fixed ±1**, fixed because the sign comes from the null `p`, not from the data.
Summing,

    d_u(x) = ‖x‖_V = Σ_{i=1}^{n−1} sgn(g_i) D_i = Λ(x)      (Eq. (5))

— every count entering with a + or − decided in advance. That is the "fixed signed
sum", and it is a **linear** functional of the counts.

**Why that matters.** A linear functional cannot be phase-blind. If a comb δ
pushes Λ up by some amount, then −δ — the same defect with the other parity raised
— pushes it down by exactly that amount, so one comb is detected and its mirror is
anti-detected. On a monotone null it is starker still: all the signs agree, the sum
telescopes, and `Λ = ±(x_n − x_1)`. Only the first and last bins survive;
everything between them cancels, so of a comb only its effect on the two end bins
is seen, with a sign that depends on which parity is raised.

"Collapses" is therefore literal. A nonlinear, phase-blind statistic degenerates
into a linear, phase-sensitive one — with no warning, and no change in how it is
computed. Centering keeps the fold: the centered counts `r = x − Np` have no
built-in slope, so their gaps `r_{i+1} − r_i = D_i − N g_i` straddle zero and `|·|`
keeps folding.

## Detected and anti-detected

Under a sloped null the uncentered `d_u` degenerates into the linear functional
`Λ(x) = Σ_{i=1}^{n−1} sgn(p_{i+1} − p_i) (x_{i+1} − x_i)` (Eq. (5), Proposition 1). **The detected comb
is whichever sign pushes Λ upward; the other pushes it down and is
anti-detected** — the upper-tail test then fires *less* often than on clean data.

What that looks like depends on the null:

- **Null already alternates** (q = 1:2:1:2:1). Its slope signs alternate, so a
  comb that **deepens** the existing zigzag raises Λ and is detected (power
  0.332); the comb that **flattens** it lowers Λ and is anti-detected (0.002,
  against a size of 0.047). Sign − raises bins 2 and 4, which are the null's high
  bins, so sign − is the detected one here.
- **Null is monotone up to a flat step** (run-length, q = 16:8:4:2:1:1). The five
  slopes are negative except the last, `g_5 = p_6 − p_5 = 0`, whose term gets
  sign 0, so Λ telescopes to `x_1 − x_5` and alternation is irrelevant — only
  bins 1 and 5 survive. Sign + raises bin 1 far more than bin 5 (0.500 → 0.576 against
  0.031 → 0.036), so `x_1 − x_5` rises and sign + is detected (0.252); sign −
  does the opposite and is anti-detected (0.041). Bin 6 drops out of Λ because
  `g_5 = 0`, not because the comb spares it.

The two nulls therefore disagree about which parity dies. **There is no parity
rule**, and none should ever be stated.

## Collapse index

    index = Corr_H0( ‖x‖_V , Λ(x) )

the Pearson correlation, under the null, between the DTV and the linear
functional it degenerates into. It needs only `p`, `N` and a few thousand null
draws — no alternative, so it can be computed before any data are collected.

Higher is worse **for the uncentered `d_u`**: at index 0 its worse-sign
power is 0.85–0.89, and it declines monotonically with the index (rank correlation
−0.90 to −0.84 between index and power, over seeds in `experiment_rho_seeds.txt`; e.g. 0.43 at index 0.73 in one
row of `experiment_rho_sweep.txt`) to at or below α near 1. There is **no threshold**; the index orders the loss.
The CT is indifferent, never below 0.57 anywhere in the range. On a uniform null
every `g_i = 0`, so Λ ≡ 0 and the correlation is 0/0, undefined; the code reports
0 by convention. (That `d_u` and the CT coincide on the flat row of Table I follows
from Eq. (3) with all `q_i` equal, `S = n d_u`, not from the index.)

In the Letter it is estimated by simulation, except in Table I, where
`experiment_exact_index.py` computes it exactly at N = 40 by enumerating the
at-most-four-bin marginals its moments need. Closed-form null moments are listed
among the Letter's future work.

## The statistics

**Uncentered DTV** `d_u` — `‖x‖_V = Σ_{i=1}^{n−1} |x_{i+1} − x_i|` (Eq. (1)), the
uniform-null comb test's statistic applied directly to the counts. Result files
call it `du` or `DTV`. On a non-uniform null it is not a test; the
Letter only analyses what it would measure (Section III).

**CT with null r / TVC** — the comb test, whose
statistic the Letter calls the centered total variation,
`d_r(x) = ‖x − Np‖_V = Σ_{i=1}^{n−1} |D_i − N g_i|` (Eq. (2)): the same
variation applied to the counts **after** subtracting what the null expects.
Centering the *counts*, not the statistic's value — subtracting a constant from a
linear functional leaves it linear and changes nothing.

**T(t) family** — `y_i = r_i / (N p_i (1 − p_i))^{t/2}`,
`T(t) = Σ_{i=1}^{n−1} |y_{i+1} − y_i|`. `T(0) = d_r`; `T(1)` is the DTV of the adjusted residuals. No member is uniformly
best: **scaled** combs favour `T(0)`, **shifted** combs favour `T(1)`, and the
power gap reaches 0.22.

**When the family collapses to one test.** Each member divides bin `i`'s centered count
by `(N p_i (1-p_i))^(t/2)`, the bin's standard deviation to the power `t`. If every
`p_i(1-p_i)` is the same number `v`, that divisor is the same constant in every bin
and factors out:

    T(t) = sum |y_{i+1} - y_i| = (Nv)^(-t/2) * sum |r_{i+1} - r_i| = (Nv)^(-t/2) T(0)

so every member is a rescaling of `T(0)`: one test, one exact null, all
lattice-valued. Since `x(1-x) = v` has at most two roots, this needs the `p_i` to
take at most two values `a` and `1-a` — in practice the uniform null (or a two-bin
null), where variance weighting has nothing to weight. Otherwise only `T(0)` is
lattice-valued and the others need the gamma or Monte Carlo.

**Scaled vs shifted** — two ways a comb can sit on a non-flat null:

- **scaled** (multiplicative), `p_i(1 ± ε)`: the departure is proportional to
  `p_i`, so it is largest where the null is largest. This is the physical
  converter model — capacitor mismatch *scales* code widths.
- **shifted** (additive), `p_i ± δ`: the same absolute size in every bin.

## min-p, Tippett, and the two batches

A practitioner cannot know whether their defect is scaled or shifted, so the
Letter recommends combining the family rather than betting on a member.

**Tippett's combination (min-p)** takes the **minimum** of the three p-values of
`T(0)`, `T(0.5)`, `T(1)` and rejects when it is small: if *any* member sees the
defect, catch it. Its worst-case shortfall is 0.03, against 0.22 for committing to
`T(0)` and 0.19 for chi-square.

**Why two batches.** A minimum of three p-values is a multiple-comparison problem:
under the null it is stochastically smaller than uniform, so it is **not itself a
p-value** and using it raw would inflate the size. It needs its own calibration.

1. **Batch 1** builds each statistic's null distribution, turning an observed
   `T(t)` into an empirical p-value.
2. **Batch 2**, independent, is pushed through batch 1's p-value maps, min-p is
   computed for each draw, and the α-quantile of those becomes the critical value.

The second batch must be independent: on batch 1 the p-values are uniform *by
construction*, so calibrating the minimum there would give an optimistic cut and
the wrong size.

**Šidák cuts** are the correction you would use if the three tests were
**independent**: `Pr[min p < c] = 1 − (1 − c)^3`, so size α needs
`c = 1 − (1 − α)^{1/3}` — exact under independence, slightly sharper than
Bonferroni's α/3. It is the wrong tool here. The three members are three
weightings of the same centered counts and their p-values correlate at 0.93–1.00, so
assuming independence makes the cut far too conservative and costs 0.09–0.15 in
power. Monte Carlo calibration uses min-p's *actual* null distribution and absorbs
whatever correlation is really present.

## Pooled and unpooled chi-square

**Unpooled X²** is Pearson's chi-square on the bins as they are. Its null
distribution is the χ²(n−1) *approximation*, which needs every expected cell count
to be reasonably large — the textbook rule is ≥ 5. On the cell and moon nulls
(n = 64, N = 2000) 22 and 26 bins have expected count below 1 (camera has none,
page 2, coins 7), so the asymptotic cut breaks there: attained size 0.053–0.123
against a nominal 0.05. **Table II's X² column is MC-calibrated**, so its size is
nominal; the 0.053–0.123 refers to the asymptotic cut (repository only; the Letter
no longer quotes it).

**Pooled X²** merges adjacent bins until each group's expected count reaches 5,
then runs chi-square on the groups. The merging uses the **null alone**, so it
cannot be tuned to the alternative. This brings the asymptotic cut back to a
correct size.

The Letter leads with the **pooled** comparison because pooling is chi-square's
practical form on a sparse histogram: the CT beats unpooled X² by 2.1–5.7× but
only 1.86–2.45× pooled; both X² columns are MC-calibrated, so both gaps are power,
not size. The cost of pooling is that merging adjacent bins destroys exactly the
fine structure a comb lives in.

*(Unrelated sense: "collecting runs of 6 or more" in Section V-C lumps the
geometric tail into one bin. That is not chi-square pooling.)*

## The dashes in Table II

`--` means the statistic was not computed in that experiment. There are three
distinct cases, and only one of them is a matter of principle:

| block | missing | why |
|-------|---------|-----|
| 1 (tent, run-length) | `X2p` | n <= 7, N = 500: expected counts ~75, **nothing to pool** |
| 3 (converter) | `X2p` | n = 64, N = 10^4: expected ~156, nothing to pool |
| 2, 3 | `CvM`, `AD` | not computed in those experiments |

So the pooled chi-square column is the images' column by necessity: pooling merges
adjacent bins until each group's expected count reaches 5, and only the image nulls
(n = 64, N = 2000; cell and moon have 22 and 26 bins below one) have anything to merge. Running it on the
other blocks would return the unpooled statistic unchanged.

The other blanks are experimental scope, not results that were run and withheld.
Nothing was omitted because it was unfavorable; `min-p` on the images (0.29 to
0.59) sits just below the CT, as Section IV-B predicts for a multiplicative comb.

## Calibration capture (the null must be known)

Everything in the Letter assumes the null `p` is **known**. If you instead
estimate it from a separate clean acquisition — a **calibration capture** — the
estimation error inflates the size of the CT, X² and T(1); the uncentered `d_u`, blind to
the slope it has locked onto, keeps a mean size of 0.040 at M0/N = 1 (though up to
0.51 on one image). `M0/N` is the length of that
capture relative to the test sample. Measured in
`results/experiment_stress_misspec.txt`, nominal size 0.05:

| M0/N | CT    | X²    |
|------|-------|-------|
| 1    | 0.650 | 0.872 |
| 5    | 0.155 | 0.214 |
| 25   | 0.066 | 0.090 |
| 125  | 0.053 | 0.064 |

At `M0/N = 1` — a capture the same length as the test sample — the tests reject
65 % and 87 % of **clean** data against a nominal 5 %; neither is usable. It takes
a capture **125 times longer** to bring them back to 0.053 and 0.064.

This is the Letter's most important caveat for a practitioner: the CT is not a
plug-in method. The null has to be known analytically (converter code density
under a sine or ramp, run lengths of a fair bit source) or exactly (each image's
own code law), not estimated from a short capture.

## Tilt

A **tilt** is a linear trend across the bins — probabilities raised at one end and
lowered at the other. The opposite extreme from a comb. It matters because,
**locally** (against a small tilt, where chi-square's local power is the same in
every direction, 0.090–0.099), the CT has essentially **no** power against it
(0.051, sitting at its size); its largest local power, 0.117–0.154, is against
an edge-damped comb; CvM and AD, weak on combs, are strong on tilts (0.58–0.88). The families are
complementary, and the Letter says so rather than claiming the CT dominates
(Section V-B; `experiment_local_sensitivity.txt`).

## Exactness machinery

**Rational null** — `p_i = q_i / L` with integer `q_i` and `L = Σ q_i`. Then
`S = L·d_r` is an **integer**, so the dynamic program runs in exact arithmetic with
no floating point anywhere.

**Why the `q_i^m` weighting is legal** — since `q_j = L p_j` and `Σ_j x_j = N`,

    Π_j q_j^{x_j} = L^{Σ x_j} Π_j p_j^{x_j} = L^N Π_j p_j^{x_j}

so the integer weight the DP accumulates for a configuration *is* its multinomial
mass, scaled by `L^N`; dividing the total by `L^N` returns a probability.

**Lattice-valued** — every value of `d_r` is a multiple of the single step `1/L`, so
the values can be indexed as integer states of a recurrence. That is what makes a
tractable exact null possible.

**Congruent, mod L** — `a ≡ b (mod L)` means `a` and `b` leave the same remainder
when divided by `L`, equivalently that `L` divides `a − b`. Clock arithmetic is the
everyday case: 17 ≡ 5 (mod 12), because 17:00 is 5 o'clock. A **residue class mod
L** is the set of all integers sharing one remainder.

**Residue classes and c** — every term of `S` (Eq. (3)) is congruent to
`± N(q_{i+1} − q_i)` mod `L`, so the reachable values lie in `c ≤ min(L, 2^{n−1})`
residue classes mod `L` and number O(cN) rather than O(LN). `c` is set by `q` and
`N` (through `N mod L`), and is 1 to 16 for the nulls in the Letter at N ≤ 100.
**The cost factor is set by `c`, not by `L`** — scaling the weights
`q → λq` leaves `c` and the support size unchanged even though it multiplies `L`;
only the larger integers cost time (`verify_dr_exact.py`, check 6, for the invariance;
`experiment_dp_complexity.txt` for the timing).

Why only `c` classes: each term is `|L(x_{i+1} − x_i) − N(q_{i+1} − q_i)|`, and the
first piece is a multiple of `L`, so it vanishes mod `L` whatever the data are.
What survives is `± N(q_{i+1} − q_i)`, the sign depending on how the absolute value
resolves — two candidate residues per term, fixed by the null; which one occurs is
fixed by the sample.
Over `n − 1` terms that is at most `2^(n−1)` combinations, and never more than `L`
remainders in all, so `S` can only land inside `c ≤ min(L, 2^(n−1))` classes and
its support is O(cN) rather than O(LN).

**Timings.** Wall-clock figures are machine-dependent and the Letter quotes none.
`results/exact_reach_timing.txt` (`experiment_timing.py --fair`) reports the ratio
of the DP for `d_r` to the DP for `d_u` run in the SAME sparse program (increment
`|m - k|`), as a median over repeats: 2.5, 2.0, and 1.3 at N = 50, 100, 200,
tracking `c` = 8, 4, 2 (Section II-A). Timed against the dense reference
`du_exact.py` instead, the ratios read 2.1, 1.5, and 0.7; a ratio below one there
reflects that program re-zeroing an (N+1)x(N+1)x(2N+2) buffer per bin, not a
cheaper DP for `d_r`; the state-count argument says c >= 1, so it never is.