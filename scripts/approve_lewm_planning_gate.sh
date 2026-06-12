#!/usr/bin/env bash
# Approve a reviewed LeWM planning-readiness report for one exact checkpoint.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/approve_lewm_planning_gate.sh <planning-gate-report.json>"
  exit 1
fi

REPORT="$1"
if [[ ! -f "$REPORT" ]]; then
  echo "Error: report not found: $REPORT"
  exit 1
fi

case "$(basename "$REPORT")" in
  planning_gate_lewm_seq*_e*.json)
    ;;
  *)
    echo "Error: expected a report named planning_gate_lewm_seq<N>_e<EPOCH>.json"
    exit 1
    ;;
esac

MARKER="${REPORT%.json}.approved"
{
  echo "approved_at=$(date --iso-8601=seconds)"
  echo "report=$REPORT"
} > "$MARKER"
echo "Approved planning-readiness gate: $MARKER"
