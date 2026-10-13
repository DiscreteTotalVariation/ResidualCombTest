#!/bin/bash
# Generate exact DTV distributions for the non-uniform weights 1, 2, 3, 4, 5, 6, 7
# (n = 7) for sample sizes N = 1..200, using du_exact.py.
# Skips any N whose output file already exists.
#
# Output: ../data/exact_distributions/1_2_3_4_5_6_7/N_<N>.txt
# Run from anywhere (cd is automatic).

set -e
cd "$(dirname "$0")"

OUTPUT_DIR="../data/exact_distributions/1_2_3_4_5_6_7"
mkdir -p "$OUTPUT_DIR"

for N in $(seq 1 200); do
    output_path="$OUTPUT_DIR/N_$N.txt"
    if [ -f "$output_path" ]; then
        continue
    fi
    python3 du_exact.py -N "$N" --probs 1 2 3 4 5 6 7 -o "$output_path"
done
