#!/usr/bin/env bash
# Full resumable mass-datagen: phase 1 rollout+convert+labels, phase 2 batched
# render. Both phases are independently resumable (per-chunk / per-scene done
# markers), so after a freeze or reboot just re-run this one script and it
# picks up where it left off. Intended to run under tmux.
set -u
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "=== PHASE 1: rollout + convert + labels ($(date -u +%FT%TZ)) ==="
bash "$SCRIPT_DIR/datagen_rollout_resumable.sh"
echo "=== PHASE 2: batched render ($(date -u +%FT%TZ)) ==="
bash "$SCRIPT_DIR/datagen_render_resumable.sh"
echo "=== DATAGEN COMPLETE ($(date -u +%FT%TZ)) ==="
