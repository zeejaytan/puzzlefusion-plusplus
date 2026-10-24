#!/usr/bin/env bash
# Render a single PuzzleFusion++ inference sample (data_id) to gt.png + video.mp4
# using the existing Blender-based renderer.
#
# Usage (on a login or GPU node with Blender available):
#   cd /data/gpfs/projects/punim2657/Puzzlefusion
#   ./render_sample.sh 97
#
# This assumes:
# - config/auto_aggl.yaml points at the correct mesh + inference paths
# - output/denoiser/everyday_epoch2000_bs64/inference/test_run/<data_id>/ exists

set -euo pipefail

DATA_ID="${1:-97}"

# Renders go to the working area, never the checkout.
OUT=${OUT:-/data/gpfs/projects/punim2657/PFpp/out/sample_${DATA_ID}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"

echo "Rendering sample data_id=${DATA_ID}"

export PYTHONUNBUFFERED=1

python -m renderer.render_results \
  experiment_name=everyday_epoch2000_bs64 \
  inference_dir=test_run \
  renderer.output_path="$OUT" \
  +renderer.single_data_id="${DATA_ID}"

