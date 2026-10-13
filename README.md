# ResidualCombTest

The comb test (CT) for histograms under a non-uniform null.

Companion repository for the letter *Comb Test Under a Non-Uniform Null:
Centered Total Variation and Phase-Blindness* (Banić & Elezović), the follow-up
to *Comb Test: Histogram Uniformity Testing Based on Discrete Total Variation*,
IEEE Signal Processing Letters, vol. 33, pp. 3726–3730, 2026,
[doi:10.1109/LSP.2026.3728406](https://doi.org/10.1109/LSP.2026.3728406).

The CT with null `r` is the discrete total variation of the centered counts,
`d_r(x) = ||x - N r||_V`.  It detects alternating ("comb") defects, every second
bin raised, whichever bins are raised.  The uncentered `d_u(x) = ||x||_V` of the
published letter would lose that property on a steep non-uniform null.

## Using the test

```bash
cd code
python3 comb_test.py --counts 8 12 8 12 0 --weights 1 2 1 2 1
python3 comb_test.py --counts 8 12 8 12 0 --probs 0.1 0.2 0.1 0.2 0.4
```

It reports `d_r`, the critical value, the p-value, a decision at `--alpha`
(default 0.05), and the collapse index of the null (0: the uncentered `d_u` would
lose nothing; 1: it would detect only one comb sign).  The null is calibrated
exactly when it is small enough and by a moment-matched gamma otherwise;
`--minp` adds the min-p combination of Section IV-B.  The null must be known, not
estimated from the data.  `python3 comb_test.py --help` lists every option;
`REPRODUCING.md` explains them.

## Reproducing the Letter

```bash
pip install -r requirements-lock.txt
bash pipeline.sh --check          # exact-DP verification (seconds)
bash pipeline.sh --plots-only     # figures and tables from the shipped results/
bash pipeline.sh --from-scratch   # rerun every experiment (about 40 min), then plots
bash pipeline.sh --stress         # large randomized and exhaustive checks (about 70 min)
```

With the versions in `requirements-lock.txt` every file in `results/`
reproduces byte for byte, except for wall-clock timings.  Every number in the
Letter traces to a line in `results/`; `REPRODUCING.md` maps each one to its
script, file and mode.

## Layout

| path | contents |
|---|---|
| `code/` | the test (`comb_test.py`), the exact null (`dr_exact.py`), checks and every experiment |
| `results/` | captured output of every script |
| `figures/`, `tables/` | the scripts that draw Fig. 1 and write Tables I and II into `paper/` |
| `data/` | precomputed exact distributions; gitignored, regenerated on demand (`data.md`) |
| `REPRODUCING.md` | notation, experiments, seeds, run times, and where each number comes from |
| `GLOSSARY.md` | the terms the Letter uses in passing |

## Provenance

The code in this repository was drafted with generative artificial intelligence
(Claude Opus 5 and Fable 5.1, Anthropic) under the authors' direction and verified
by the authors against brute-force enumeration and the checks in `pipeline.sh
--check`; the authors take responsibility for all of it.

## Citation

If you use this code, please cite the parent paper, whose exact dynamic
program this code extends:

```bibtex
@article{banic2026combtest,
  title   = {Comb Test: Histogram Uniformity Testing Based on Discrete Total Variation},
  author  = {Bani\'c, Nikola and Elezovi\'c, Neven},
  journal = {IEEE Signal Processing Letters},
  volume  = {33},
  pages   = {3726--3730},
  year    = {2026},
  doi     = {10.1109/LSP.2026.3728406}
}
```
