#!/bin/bash
# Extract evaluation metrics from a PF++ test.py log into a compact summary.
# Usage:  ./extract_eval_metrics.sh <log_file>

LOG="${1:-/data/gpfs/projects/punim2657/Puzzlefusion/logs/eval_oracle_23959887.out}"

if [ ! -f "$LOG" ]; then
    echo "ERROR: log file not found: $LOG" >&2
    exit 1
fi

echo "=== Extracting metrics from $LOG ==="
echo ""
# Each dataset is delimited by "══ <name> ══" and followed by 4 eval/... lines
awk '
/══/ { name=$0; gsub(/═+/, "", name); gsub(/^ +| +$/, "", name); if (NR > 1) print ""; print "── " name " ──"; next }
/eval\/part_acc/ { printf "  part_acc:  %s\n", $NF }
/eval\/rmse_r/   { printf "  rmse_r:    %s\n", $NF }
/eval\/rmse_t/   { printf "  rmse_t:    %s\n", $NF }
/eval\/shape_cd/ { printf "  shape_cd:  %s\n", $NF }
/Traceback|Error|FAILED/ { printf "  [!] %s\n", $0 }
' "$LOG"
