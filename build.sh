#!/bin/bash
# Build the Docker image for the comb test (CT) pipeline.
set -e
cd "$(dirname "$0")"
docker build -t residualcombtest .
echo "Docker image 'residualcombtest' built successfully."
