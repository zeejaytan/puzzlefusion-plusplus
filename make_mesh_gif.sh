#!/usr/bin/env bash
# Generate a mesh-based assembly GIF for a single PuzzleFusion++ sample.
#
# Usage (on Spartan):
#   cd /data/gpfs/projects/punim2657/Puzzlefusion
#   ./make_mesh_gif.sh 97
#
# This:
#   1) runs the existing Blender renderer (renderer/render_results.py) for one sample
#      by setting renderer.num_samples=1 (deterministic sampling, seed fixed inside code)
#   2) converts the resulting video.mp4 into mesh_assembly_<id>.gif using ffmpeg
#
# NOTE: the numeric argument is used only to name the GIF; the actual sample ID
# rendered is the first one selected by MyRenderer.sample_data_files().

set -euo pipefail

DATA_ID="${1:-97}"
PROJDIR="/data/gpfs/projects/punim2657/Puzzlefusion"

cd "$PROJDIR"

echo "=== Rendering mesh animation with Blender ==="
echo "Project: $PROJDIR"
echo "GIF label (data_id for naming only): $DATA_ID"

module load Anaconda3/2024.02-1 || true
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp

# Run the Hydra-based renderer, limiting to a single sampled item.
python -m renderer.render_results \
  experiment_name=everyday_epoch2000_bs64 \
  inference_dir=test_run \
  renderer.output_path="gif_run" \
  renderer.num_samples=1

LATEST_DIR=$(ls -td "$PROJDIR"/BlenderToolBox_render/gif_run/* 2>/dev/null | head -n 1 || true)
if [[ -z "$LATEST_DIR" ]]; then
  echo "ERROR: Could not find BlenderToolBox_render/gif_run/* directory."
  exit 1
fi

VIDEO="$LATEST_DIR/video.mp4"
if [[ ! -f "$VIDEO" ]]; then
  echo "ERROR: Expected video.mp4 not found at $VIDEO"
  exit 1
fi

echo "=== Converting video to GIF ==="
OUT_GIF="$PROJDIR/mesh_assembly_${DATA_ID}.gif"

ffmpeg -y -i "$VIDEO" -vf "fps=20,scale=640:-1:flags=lanczos" "$OUT_GIF"

echo "Wrote GIF: $OUT_GIF"

