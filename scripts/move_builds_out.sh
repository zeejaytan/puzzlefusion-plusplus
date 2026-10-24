#!/usr/bin/env bash
# Move the compiled pytorch3d and chamferdist out of the Spartan checkout into the
# working area, and point the puzzlefusionpp environment at the new place.
# Workbench-hygiene ticket 01. Moving, not rebuilding: pytorch3d takes an hour to compile.
#
# Usage (Spartan login node):
#   scripts/move_builds_out.sh            # dry run: guards, plan, files that would change
#   scripts/move_builds_out.sh --apply    # do it, then check the imports on this node
#
# Nothing is deleted. Every environment file it edits is copied first to
# $WORK/attic/env-backup-<stamp>/, and the undo commands are printed at the end.
set -euo pipefail

P=/data/gpfs/projects/punim2657/Puzzlefusion
WORK=/data/gpfs/projects/punim2657/PFpp
BUILD=$WORK/build
APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1
die() { echo "REFUSED: $*" >&2; exit 1; }

module load Anaconda3/2024.02-1 >/dev/null 2>&1
eval "$(conda shell.bash hook)"
conda activate puzzlefusionpp
SP=$(python -c 'import site; print(site.getsitepackages()[0])')
echo "env:           $CONDA_PREFIX"
echo "site-packages: $SP"

# Guards: no job running from the checkout, nothing written in the last hour.
jobs=$(squeue -h -u "$USER" -o '%i %Z' | awk -v p="$P" 'index($2, p) == 1' || true)
[[ -z "$jobs" ]] || die "jobs still have their WorkDir in $P: $jobs"
for d in pytorch3d chamferdist; do
    [[ -d "$P/$d" ]] || die "$P/$d does not exist"
    [[ ! -e "$BUILD/$d" ]] || die "$BUILD/$d already exists"
    recent=$(find "$P/$d" -newermt '1 hour ago' -print -quit)
    [[ -z "$recent" ]] || die "$P/$d written in the last hour: $recent"
done
[[ $(stat -c %d "$P") == $(stat -c %d "$(dirname "$WORK")") ]] \
    || die "working area is on another filesystem; mv would copy"

# Environment files that name the old paths (.pth, .egg-link, editable finders, pip metadata).
mapfile -t refs < <(grep -rlE "$P/(pytorch3d|chamferdist)" \
    "$SP"/*.pth "$SP"/*.egg-link "$SP"/__editable__* "$SP"/*.dist-info/direct_url.json \
    "$SP"/*.dist-info/RECORD "$SP"/*.egg-info 2>/dev/null || true)
echo "files naming the old paths: ${#refs[@]}"
for f in "${refs[@]}"; do echo "  $f"; grep -hoE "$P/(pytorch3d|chamferdist)[^\"' ]*" "$f" | sort -u | sed 's/^/      /'; done

if (( ! APPLY )); then
    echo "DRY RUN: would mv $P/{pytorch3d,chamferdist} -> $BUILD/ and rewrite the files above."
    exit 0
fi

stamp=$(date +%Y%m%d-%H%M%S)
bak=$WORK/attic/env-backup-$stamp
mkdir -p "$BUILD" "$WORK/logs" "$WORK/out" "$WORK/attic" "$WORK/runs" "$bak"
[[ -e "$WORK/runs/index.tsv" ]] || printf 'runid\tjobid\tcommit\ttag\tverdict\thuman_note\n' > "$WORK/runs/index.tsv"

for f in "${refs[@]}"; do cp -a --parents "$f" "$bak/"; done
mv "$P/pytorch3d" "$BUILD/pytorch3d"
mv "$P/chamferdist" "$BUILD/chamferdist"
for f in "${refs[@]}"; do
    sed -i -e "s#$P/pytorch3d#$BUILD/pytorch3d#g" -e "s#$P/chamferdist#$BUILD/chamferdist#g" "$f"
done
{
    echo "moved $stamp by scripts/move_builds_out.sh (puzzlefusion-plusplus ticket 01)"
    echo "$P/pytorch3d -> $BUILD/pytorch3d"
    echo "$P/chamferdist -> $BUILD/chamferdist"
    printf 'edited (originals in %s):\n' "$bak"; printf '  %s\n' "${refs[@]}"
} > "$WORK/attic/move-builds-$stamp.txt"

echo "=== import check on $(hostname) ==="
cd "$WORK"
python - "$BUILD" <<'EOF'
import sys, pytorch3d, chamferdist
build = sys.argv[1]
print("pytorch3d  ", pytorch3d.__version__, pytorch3d.__file__)
print("chamferdist", chamferdist.__file__)
assert pytorch3d.__file__.startswith(build), "pytorch3d is not loaded from the new place"
from pytorch3d import _C  # the compiled extension, inside the moved tree
print("pytorch3d._C", _C.__file__)
EOF
echo "OK. Undo: mv $BUILD/pytorch3d $BUILD/chamferdist $P/ && cp -a $bak$SP/. $SP/"
