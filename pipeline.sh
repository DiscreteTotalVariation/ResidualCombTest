#!/bin/bash
# Reproduce the letter's numbers, tables and figures.
#
#   bash pipeline.sh --plots-only     regenerate paper/img/*.pdf and paper/table*.tex
#                                     from the committed results/ (under a minute)
#   bash pipeline.sh --tables-only    regenerate paper/table*.tex only
#   bash pipeline.sh --from-scratch   rerun every experiment into results/, then plots
#                                     (about one hour; the run-length experiment alone
#                                     takes ~10 min and the exact DP at N=200 ~90 s)
#   bash pipeline.sh --check          run the exact-DP verification and the
#                                     determinism check only
#   bash pipeline.sh --stress         large-scale randomized stress tests of the
#                                     Letter's near-universal claims (~75 min:
#                                     ~50 min for the stress, robustness and design
#                                     parts, ~13 min for the bit-source lane tests,
#                                     ~12 min for the first-order rate on 16 cores);
#                                     not part of --from-scratch, and run after it
#                                     (the lane tests compare against its
#                                     random-lane run-length files).  The numbers
#                                     the Letter marks "(repository)" come from
#                                     both modes; REPRODUCING.md, "Where each
#                                     (repository) number comes from", maps every
#                                     one of them to its file, lines and mode
#
# Every script uses a fixed seed.  The deterministic scripts (verify_dr_exact.py,
# experiment_exact_power.py, experiment_exact_index.py) must reproduce their
# results files byte for byte; the Monte Carlo scripts reproduce them exactly with
# the same numpy version and within Monte Carlo error otherwise.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONWARNINGS=ignore
mode="${1:---plots-only}"

check() {
  (cd code && python3 verify_dr_exact.py | tail -1)
  (cd code && python3 verify_comb_test.py | tail -1 | sed 's/^/comb_test.py self-test -- /')
  tmp=$(mktemp -t dr_exact_XXXXXX)
  (cd code && python3 experiment_exact_power.py > "$tmp" && diff -q "$tmp" ../results/experiment_exact_power.txt >/dev/null 2>&1 \
     && echo "experiment_exact_power.py reproduces results/ byte for byte" \
     || echo "WARNING: experiment_exact_power.py differs from results/ (re-run --from-scratch or inspect)")
  rm -f "$tmp"
}

experiments() {
  cd code
  for s in verify_dr_exact experiment_exact_power experiment_exact_index experiment_dp_complexity \
           experiment_null_approximation experiment_gamma_large_n \
           experiment_degeneracy experiment_rho_diagnostic experiment_rho_sweep experiment_bias \
           experiment_selection_rule \
           experiment_shape_power experiment_local_sensitivity \
           experiment_variants experiment_double_jpeg experiment_sine_histogram_adc; do
    echo "### $s"; python3 $s.py > ../results/$s.txt 2>&1 || echo "FAILED: $s"
  done
  echo "### experiment_runlength_rng (full, ~10 min)"
  python3 experiment_runlength_rng.py > ../results/experiment_runlength_rng.txt 2>&1 || echo "FAILED: runlength"
  python3 experiment_runlength_rng.py --skip_b --N 200 1000 --exact_N 100 200 --bits 8000000 \
      > ../results/experiment_runlength_rng_partA_exact.txt 2>&1 || echo "FAILED: runlength partA"
  python3 experiment_runlength_rng.py --skip_a --bits 4000000 --window_N 1000 2500 --random_phase \
      --eps 0 0.05 0.07 0.10 > ../results/experiment_runlength_rng_randomphase_5000bits.txt 2>&1 || echo "FAILED: runlength 5000bits"
  # The lane phase matters: held fixed it aligns the longest-run test's 8-bit
  # blocks with the defect, which is that test's most favourable case.  Both are
  # produced so the Letter can report the realistic model and disclose the other.
  python3 experiment_runlength_rng.py --skip_a --random_phase \
      > ../results/experiment_runlength_rng_randomphase.txt 2>&1 || echo "FAILED: runlength randomphase"
  echo "### per-image real-image run"
  for img in camera coins moon page cell; do echo "### $img"; python3 experiment_real_images.py --images $img --N 2000 --eps 0 0.06 -0.06 0.10 -0.10; done \
      > ../results/experiment_real_images_per_image.txt 2>&1 || echo "FAILED: per-image"
  python3 experiment_real_images.py > ../results/experiment_real_images_phase.txt 2>&1 || echo "FAILED: images phase"
  python3 experiment_selection_rule.py --exponents 0 0.25 0.5 0.75 1 > ../results/experiment_selection_rule_t5.txt 2>&1 || echo "FAILED: selection t5"
  python3 experiment_rho_diagnostic.py --distance 0.04 > ../results/experiment_rho_diagnostic_l1_0.04.txt 2>&1 || echo "FAILED: rho 0.04"
  python3 experiment_rho_diagnostic.py --distance 0.16 > ../results/experiment_rho_diagnostic_l1_0.16.txt 2>&1 || echo "FAILED: rho 0.16"
  python3 check_minp.py --part sidak > ../results/check_minp_sidak.txt 2>&1 || echo "FAILED: check_minp_sidak"
  python3 check_minp.py --part phase > ../results/check_minp_phase.txt 2>&1 || echo "FAILED: check_minp_phase"
  python3 experiment_sine_histogram_adc.py --n 256 --N 2000 10000 50000 --eps 0 0.05 -0.05 0.10 -0.10 \
      > ../results/experiment_sine_histogram_adc_n256.txt 2>&1 || echo "FAILED: experiment_sine_histogram_adc_n256"
  # IEEE 1241 row with the arcsine null fitted from each record (~3 min, seed 33).
  echo "### experiment_sine_fitted_null"
  python3 experiment_sine_fitted_null.py \
      > ../results/experiment_sine_fitted_null.txt 2>&1 || echo "FAILED: sine_fitted_null"
  # Period-2 and period-4 combs, CT against AC, X^2 and min-p(AC, X^2) (~12 s per
  # run).  Default seeds 19/16 are the parent scripts' (draw-for-draw reproduction of
  # the per-image and acperiod files, so it runs after the per-image run above; the
  # acperiod file comes from --stress or the shipped results/); the second run is
  # the seed check on independent streams.
  echo "### experiment_period_competitors"
  python3 experiment_period_competitors.py \
      > ../results/experiment_period_competitors.txt 2>&1 || echo "FAILED: period_competitors"
  python3 experiment_period_competitors.py --seed_images 1019 --seed_synthetic 1016 \
      > ../results/experiment_period_competitors_seedcheck.txt 2>&1 || echo "FAILED: period_competitors seedcheck"
  cd ..
}

tables() { python3 tables/table1.py >/dev/null; python3 tables/table2.py >/dev/null; echo "tables written to paper/"; }
plots()  { for f in figures/fig*.py; do python3 "$f"; done; }

case "$mode" in
  --stress)       (cd code
                   # Case counts are pinned here, NOT left to the script default.
                   # The randomized parts were once run ad hoc at 150/120 cases and
                   # those stronger results were captured; a later default-valued
                   # rerun silently overwrote them with 60-case runs.  Pinning the
                   # counts in the pipeline is what makes the stronger evidence
                   # reproducible instead of a one-off.
                   cases_for() {
                     case "$1" in
                       exact)                      echo 150 ;;
                       altpath|printed|pvalue|cbound) echo 120 ;;
                       *)                          echo 60  ;;
                     esac
                   }
                   for p in exact bias phase gamma images altpath edge cbound \
                            pvalue printed levels indexn local misspec; do
                     echo "### stress: $p (cases $(cases_for $p))"
                     python3 experiment_stress.py --part $p --cases "$(cases_for $p)" \
                       > ../results/experiment_stress_$p.txt 2>&1 || echo "FAILED: $p"
                   done
                   echo "### collapse-index sweep across seeds"
                   python3 experiment_rho_seeds.py \
                     > ../results/experiment_rho_seeds.txt 2>&1 || echo "FAILED: rho_seeds"
                   echo "### exact DP timing and reach"
                   python3 experiment_timing.py --fair \
                     > ../results/exact_reach_timing.txt 2>&1 || echo "FAILED: timing"
                   echo "### exhaustive bias survey (seed-free)"
                   python3 experiment_bias_exhaustive.py \
                     > ../results/experiment_bias_exhaustive.txt 2>&1 || echo "FAILED: bias_exhaustive"
                   echo "### Proposition 1, exact"
                   python3 verify_proposition1.py \
                     > ../results/verify_proposition1.txt 2>&1 || echo "FAILED: proposition1"
                   for p in discrete twosided gammaexact drift samplesize local; do
                     echo "### robustness: $p"
                     python3 experiment_robustness.py --part $p \
                       > ../results/experiment_robustness_$p.txt 2>&1 || echo "FAILED: $p"
                   done
                   for p in zerobin acperiod codes pooling; do
                     echo "### design: $p"
                     python3 experiment_design.py --part $p \
                       > ../results/experiment_design_$p.txt 2>&1 || echo "FAILED: $p"
                   done
                   echo "### smooth"
                   python3 experiment_smooth.py > ../results/experiment_smooth.txt 2>&1 \
                     || echo "FAILED: smooth"
                   echo "### lane-aware tests on the two-lane bit source (~13 min, seed 33)"
                   python3 experiment_bitsource_lanetests.py \
                     > ../results/experiment_bitsource_lanetests.txt 2>&1 || echo "FAILED: bitsource_lanetests"
                   echo "### first-order rejection rate, exact enumeration (~12 min on 16 cores, no seed)"
                   # writes results/experiment_first_order_rate.txt itself (--output)
                   python3 experiment_first_order_rate.py --jobs 16 \
                     > /dev/null || echo "FAILED: first_order_rate") ;;
  --check)        check ;;
  --tables-only)  tables ;;
  --plots-only)   tables; plots ;;
  --from-scratch) check; experiments; tables; plots ;;
  *) echo "usage: $0 [--plots-only|--tables-only|--from-scratch|--check|--stress]"; exit 2 ;;
esac
