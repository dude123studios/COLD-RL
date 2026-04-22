#!/bin/bash
# Show experiment status: completed, running, pending.
#
# Usage: ./scripts/status.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_FILE="$REPO_DIR/results/results.json"
REGISTRY_FILE="$REPO_DIR/orchestration/experiment_registry.yaml"

# All experiment IDs from registry
ALL_IDS=$(python3 -c "
import yaml
with open('$REGISTRY_FILE') as f:
    reg = yaml.safe_load(f)
for e in reg['experiments']:
    print(e['id'])
")

# Completed IDs from results.json
if [[ -f "$RESULTS_FILE" ]]; then
    COMPLETED_IDS=$(python3 -c "
import json
completed = set()
with open('$RESULTS_FILE') as f:
    for line in f:
        line = line.strip()
        if line:
            completed.add(json.loads(line)['experiment_id'])
for eid in sorted(completed):
    print(eid)
")
else
    COMPLETED_IDS=""
fi

TOTAL=$(echo "$ALL_IDS" | wc -l)
DONE=$(echo "$COMPLETED_IDS" | grep -c . 2>/dev/null || echo 0)

echo "========================================"
echo "  HugSim Experiment Status"
echo "  Completed: $DONE / $TOTAL"
echo "========================================"
echo ""
echo "COMPLETED:"
while IFS= read -r eid; do
    if echo "$COMPLETED_IDS" | grep -q "^$eid$"; then
        echo "  [x] $eid"
    fi
done <<< "$ALL_IDS"

echo ""
echo "PENDING:"
while IFS= read -r eid; do
    if ! echo "$COMPLETED_IDS" | grep -q "^$eid$"; then
        echo "  [ ] $eid"
    fi
done <<< "$ALL_IDS"
