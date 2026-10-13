#!/bin/bash
# Generate exact CIRCULAR DTV distributions for the non-uniform weights
# 1, 2, 1, 2, 1, 2, 1, 2 (n = 8) for sample sizes N = 1..200, using circular_du_exact.py.
# Skips any N whose output file already exists.
#
# Output: ../data/exact_distributions_circular/1_2_1_2_1_2_1_2/N_<N>.txt
# Run from anywhere (cd is automatic).

set -e
cd "$(dirname "$0")"

OUTPUT_DIR="../data/exact_distributions_circular/1_2_1_2_1_2_1_2"
mkdir -p "$OUTPUT_DIR"

for N in $(seq 1 200); do
    output_path="$OUTPUT_DIR/N_$N.txt"
    if [ -f "$output_path" ]; then
        continue
    fi
    python3 circular_du_exact.py -N "$N" --probs 1 2 1 2 1 2 1 2 -o "$output_path"
done
