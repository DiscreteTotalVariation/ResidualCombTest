#!/bin/bash
# Generate exact DTV distributions for the non-uniform weights 4, 5, 4, 5, 4, 5, 4
# (n = 7) for sample sizes N = 1..200, using du_exact.py.
# Skips any N whose output file already exists.
#
# Output: ../data/exact_distributions/4_5_4_5_4_5_4/N_<N>.txt
# Run from anywhere (cd is automatic).

set -e
cd "$(dirname "$0")"

OUTPUT_DIR="../data/exact_distributions/4_5_4_5_4_5_4"
mkdir -p "$OUTPUT_DIR"

for N in $(seq 1 200); do
    output_path="$OUTPUT_DIR/N_$N.txt"
    if [ -f "$output_path" ]; then
        continue
    fi
    python3 du_exact.py -N "$N" --probs 4 5 4 5 4 5 4 -o "$output_path"
done
