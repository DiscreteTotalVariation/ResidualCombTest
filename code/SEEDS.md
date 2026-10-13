# Monte Carlo seeds

Every seed in the table below is `13 + i`, where `i` was the script's position in
the alphabetical list of scripts that used randomness when the numbering was
fixed; the seeds are frozen, so some later file names no longer sort in that
order (later scripts: see the last section).  One documented root (13), one independent
stream per experiment: a single shared seed would make scripts that draw the
same quantity reuse the same draws, so agreement between experiments would be
partly circular.  Every seed is also a `--seed` argument, so any result can be
re-run on a different stream -- except `dr_demo.py` and `verify_dr_exact.py`,
which take no command-line arguments and fix their seeds (31, 32) in the source.
`du_mc.py` is a utility rather than an experiment: it is outside the numbering
and its `--seed` defaults to 13.  So is `comb_test.py`, the user-facing test, whose
`--seed` also defaults to 13; it shares no result file with any experiment, so the
coincidence with `check_minp.py` is harmless.  `verify_comb_test.py` fixes small
seeds in the source (0, 3, 5, 7, 11, 12, 13), one per test group, and
`experiment_exact_index.py` draws nothing: its collapse indices are exact.

| seed | script |
|---|---|
| 13 | `check_minp.py` |
| 14 | `experiment_bias.py` |
| 15 | `experiment_degeneracy.py` |
| 16 | `experiment_design.py` |
| 17 | `experiment_double_jpeg.py` |
| 18 | `experiment_local_sensitivity.py` |
| 19 | `experiment_real_images.py` |
| 20 | `experiment_gamma_large_n.py` |
| 21 | `experiment_rho_diagnostic.py` |
| 22 | `experiment_rho_sweep.py` |
| 23 | `experiment_robustness.py` |
| 24 | `experiment_runlength_rng.py` |
| 25 | `experiment_selection_rule.py` |
| 26 | `experiment_shape_power.py` |
| 27 | `experiment_sine_histogram_adc.py` |
| 28 | `experiment_smooth.py` |
| 29 | `experiment_stress.py` |
| 30 | `experiment_variants.py` |
| 31 | `dr_demo.py` |
| 32 | `verify_dr_exact.py` |

## Scripts added after the numbering was fixed

Inserting a new script into the alphabetical list would renumber every script
after it and change the seed, and hence the shipped results, of experiments that
did not change.  The later scripts therefore stay outside the `13 + i` rule:

| seed | script | why |
|---|---|---|
| 19 and 16 | `experiment_period_competitors.py` | by design the PARENT scripts' seeds (`--seed_images 19` = `experiment_real_images.py`, `--seed_synthetic 16` = `experiment_design.py`), drawn in the same order, so its period-2 image rows and its `acperiod` rows reproduce those result files draw for draw; every new draw comes from a child stream `default_rng([seed, k])`.  `experiment_period_competitors_seedcheck.txt` is the same run on 1019 and 1016 |
| 33 | `experiment_bitsource_lanetests.py` | the first value after the 13..32 block |
| 33 | `experiment_sine_fitted_null.py` | likewise the first value after the block |
| none | `experiment_first_order_rate.py` | draws nothing: exact enumeration |

`experiment_bitsource_lanetests.py` and `experiment_sine_fitted_null.py` share
the value 33.  That is harmless: they simulate different quantities (bit windows
of the two-lane source; multinomial code histograms of a sine-wave test), share no
draws and no result file, and neither result is compared with the other.  Both
result files were produced with 33, so neither seed is to be changed.
