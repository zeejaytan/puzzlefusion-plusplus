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

echo "Rendering sample data_id=${DATA_ID}"

export PYTHONUNBUFFERED=1

python -m renderer.render_results \
  experiment_name=everyday_epoch2000_bs64 \
  inference_dir=test_run \
  renderer.output_path="sample_${DATA_ID}" \
  +renderer.single_data_id="${DATA_ID}"

