#!/bin/bash
# Run the pipeline inside Docker; results/ and paper/ are bind-mounted so output
# persists on the host.  Usage: ./pipeline_docker.sh [--plots-only|--from-scratch|--check]
set -e
cd "$(dirname "$0")"
IMAGE="residualcombtest"
docker image inspect "$IMAGE" >/dev/null 2>&1 || bash build.sh
mkdir -p results paper/img
docker run --rm -v "$(pwd)/results:/workspace/results" -v "$(pwd)/paper:/workspace/paper" -v "$(pwd)/data:/workspace/data" "$IMAGE" "$@"
[ "$(id -u)" -ne 0 ] && sudo chown -R "$(id -u):$(id -g)" results/ paper/ 2>/dev/null || true
echo "Done. Results in results/, figures and table bodies in paper/."
