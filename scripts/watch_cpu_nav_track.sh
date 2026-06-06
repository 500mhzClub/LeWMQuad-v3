#!/usr/bin/env bash
# Per-checkpoint CPU eval suite for the 8k-session scaling run. Runs the FULL
# suite (we have spare CPU and ~4 h between checkpoints) and appends interpreted
# findings to an ongoing markdown report. All on CPU so it never competes with
# the GPU training run.
#   PRIMARY (experiment-doc questions): rollout decomposition + MPC proxy
#   DEMO   (nice-to-have): bare clean-planner nav + energy-head pure-perception nav
set -u
ROOT=/home/andrewknowles/Workspace/LeWMQuad-v3
PY="$ROOT/.generated/venvs/genesis_render_vulkan/bin/python"
DIR="$ROOT/models/checkpoints_textured_v03_rollout_stage2_20260604/seq11_rollout_lam0p25_h10_warm2_sess8k_ep12"
TLOG=/tmp/train_sess8k.log
REPORT="$DIR/scaling_report.md"
TSV="$DIR/cpu_scaling_track.tsv"
STATE="$DIR/.eval_state.json"
RR=.generated/datagen_full
RT=.generated/datagen_full/render_textured_v03
cd "$ROOT" || exit 1

if [ ! -f "$REPORT" ]; then
  cat > "$REPORT" <<HDR
# Scaling-run ongoing eval report
**Run:** seq11, warm2, λ0.25, **8,000 sessions (~12% of the textured set, 4× the prior 2.8% model)**, 12 epochs.
Started $(date). Each checkpoint is evaluated on **CPU** (never competes with GPU training).

**Bigger questions (experiment doc):** does action-conditioned multi-second rollout prediction keep scaling?
\`zero−free\` (real action beats zero), \`free/persistence\` (beats "nothing changes"), \`free/TF\` (compounding),
and the closed-loop MPC win-rate.
**Demo (nice-to-have):** bare clean-planner beacon nav, and energy-head pure-perception nav (no heuristics).
Reference = the 2.8%-data **e3** model: \`zero−free\`@h10 +0.106, free/pers 0.59, nav −0.286 m / 0 %.
---
HDR
fi

eval_one () {
  local ck="$1"; local name ep flag dec mpc nL2 nH head trc evc log esum
  name=$(basename "$ck" .pt); ep=${name#lewm_seq11_e}
  flag="$DIR/done_${name}.flag"; [ -f "$flag" ] && return
  sleep 25  # let the checkpoint finish writing
  dec="$DIR/decomp_${name}.json"; mpc="$DIR/mpc_${name}.json"
  nL2="$DIR/navL2_${name}.json"; nH="$DIR/navH_${name}.json"; head="$DIR/head_${name}.pt"
  log="$DIR/eval_${name}.log"; : > "$log"
  echo "=== $(date) evaluating $name (CPU full suite) ===" >> "$log"

  # 1) rollout decomposition (bigger question)
  [ -f "$dec" ] || timeout 2700 "$PY" scripts/probe_lewm_rollout_horizons.py --checkpoint "$ck" \
     --data-root "$RR" --render-root "$RT" --allow-material-color-render \
     --horizons 1,2,3,5,8,10 --device cpu --output "$dec" >>"$log" 2>&1 || true
  # 2) MPC proxy (bigger question, closed-loop action preference)
  [ -f "$mpc" ] || timeout 2700 "$PY" scripts/probe_lewm_receding_mpc_proxy.py --checkpoint "$ck" \
     --data-root "$RR" --render-root "$RT" --allow-material-color-render \
     --horizons 1,2,3,5 --replan-steps 4 --batch-size 16 --max-batches 16 \
     --device cpu --output "$mpc" >>"$log" 2>&1 || true
  # 3) bare clean-planner nav (demo)
  [ -f "$nL2" ] || timeout 1800 "$PY" scripts/benchmark_lewm_closed_loop_mpc.py --checkpoint "$ck" \
     --task visible-beacon --mode kinematic --family open_obstacle_field --split test_id \
     --scene-limit 8 --policies lewm,bearing,hold,random --horizon 3 \
     --model-device cpu --backend cpu --output "$nL2" >>"$log" 2>&1 || true
  # 4) energy-head pure-perception nav (demo): cache -> train head -> nav
  if [ ! -f "$nH" ]; then
    trc="$DIR/cache_${name}_tr.pt"; evc="$DIR/cache_${name}_ev.pt"
    [ -f "$trc" ] || timeout 2700 "$PY" scripts/cache_lewm_latents.py --checkpoint "$ck" \
       --data-root "$RR" --render-root "$RT" --allow-material-color-render --seq-len 11 --stride 5 \
       --holdout-role train --max-sessions 200 --max-windows 2048 --device cpu --output "$trc" >>"$log" 2>&1 || true
    [ -f "$evc" ] || timeout 1500 "$PY" scripts/cache_lewm_latents.py --checkpoint "$ck" \
       --data-root "$RR" --render-root "$RT" --allow-material-color-render --seq-len 11 --stride 5 \
       --holdout-role eval --max-windows 512 --device cpu --output "$evc" >>"$log" 2>&1 || true
    if [ -f "$trc" ] && [ -f "$evc" ]; then
      [ -f "$head" ] || timeout 1200 "$PY" scripts/train_lewm_energy_head.py --checkpoint "$ck" \
         --train-cache "$trc" --eval-cache "$evc" --horizons 3,5,8,10 --epochs 30 \
         --device cpu --output "$head" >>"$log" 2>&1 || true
      [ -f "$head" ] && timeout 1800 "$PY" scripts/benchmark_lewm_closed_loop_mpc.py --checkpoint "$ck" \
         --head-ckpt "$head" --task visible-beacon --mode kinematic --family open_obstacle_field \
         --split test_id --scene-limit 8 --policies lewm,bearing,hold,random --horizon 3 \
         --model-device cpu --backend cpu --output "$nH" >>"$log" 2>&1 || true
    fi
    rm -f "$trc" "$evc" 2>/dev/null || true  # caches are large; drop after head training
  fi

  esum=$(grep -aE "Epoch ${ep} summary" "$TLOG" 2>/dev/null | tail -1 | sed -E 's/.*summary: //' | cut -c1-220)
  "$PY" scripts/eval_report_append.py --name "$name" --decomp "$dec" --mpc "$mpc" \
     --nav-l2 "$nL2" --nav-head "$nH" --head-ckpt "$head" \
     --report "$REPORT" --tsv "$TSV" --state "$STATE" --epoch-summary "$esum" >>"$log" 2>&1
  touch "$flag"
  echo "=== $(date) done $name ===" >> "$log"
}

while true; do
  for ck in $(ls "$DIR"/lewm_seq11_e*.pt 2>/dev/null | sort -V); do
    eval_one "$ck"
  done
  pgrep -f 'train_lewm.py' >/dev/null 2>&1 || { echo "training ended $(date); watcher exit" >> "$DIR/watcher.log"; break; }
  sleep 900
done
