#!/usr/bin/env bash
# Per-checkpoint CPU eval suite for the pose-aux fine-tune (from e3). Runs on each
# epoch checkpoint, on CPU so it never competes with the GPU fine-tune. Two groups:
#   PRIOR (must not regress): rollout decomposition (zero-free / persistence / TF),
#     MPC win-rate, bare-L2 nav, energy-head recognition nav.
#   GEOMETRY (the new objective): physical-pose RelPoseHead decode quality,
#     predictor-to-goal geometry, first-action ranking, and metric nav with no
#     privileged runtime pose.
set -u
ROOT=/home/andrewknowles/Workspace/LeWMQuad-v3
PY="$ROOT/.generated/venvs/genesis_render_vulkan/bin/python"
# Cells to sweep. FT_DIRS (space-separated) lets one watcher cover the whole
# geometry ladder (F0/C1/C2/C0-posthoc); FT_DIR stays as the single-dir default.
DIRS="${FT_DIRS:-${FT_DIR:-$ROOT/models/checkpoints_pose_aux_ft_20260606/pose_aux_selected_e3}}"
RR=.generated/datagen_full
RT=.generated/datagen_full/render_textured_v03
cd "$ROOT" || exit 1

# Per-cell report header. DIR/REPORT are set per-dir in the main loop below so a
# single watcher can sweep every ladder cell; eval_one reads those globals.
ensure_report_header () {
  local rpt="$1"
  [ -f "$rpt" ] && return
  cat > "$rpt" <<HDR
# Pose-aux geometry ladder — per-checkpoint eval report
Started $(date). All evals on CPU (never competes with the GPU training cells).
**Goal:** make the latent metric enough for navigation without privileged runtime
pose, while prior prediction metrics stay healthy. Reference (e3, pre-aux):
zero-free@h10 +0.201, free/pers 0.39, MPC vs0 0.66; frozen z_proj distance
decodability pearson ~0.05.

| epoch | zero-free@10 | free/pers@10 | MPC vs0 | navL2 m | energy nav m | **encoded xy m** | **encoded ρ** | **pred→goal xy m** | **pred→goal ρ** | **pose first ρ** | **pose first regret m** | **lewm_pose nav m** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
HDR
}

eval_one () {
  local ck="$1" name ep posehead dec mpc nL2 trc evc ehead nH pg nPose nPoseCost log flag
  name=$(basename "$ck" .pt); ep=${name#lewm_seq11_e}
  posehead="$DIR/posehead_seq11_e${ep}.pt"
  flag="$DIR/fteval_done_${name}.flag"; [ -f "$flag" ] && return
  [ -f "$posehead" ] || return         # wait until the epoch's pose head is also written
  sleep 25                             # let checkpoint finish writing
  log="$DIR/fteval_${name}.log"; : > "$log"
  echo "=== $(date) eval $name ===" >> "$log"
  dec="$DIR/decomp_${name}.json"; mpc="$DIR/mpc_${name}.json"; nL2="$DIR/navL2_${name}.json"
  trc="$DIR/cache_${name}_tr.pt"; evc="$DIR/cache_${name}_ev_pose.pt"
  ehead="$DIR/ehead_${name}.pt"; nH="$DIR/navH_${name}.json"
  pg="$DIR/posegeo_${name}.json"; nPose="$DIR/navPose_${name}.json"
  nPoseCost="$DIR/navPoseCost_${name}.json"

  # ---- caches (shared by energy-head + geometry probe) ----
  [ -f "$evc" ] || timeout 1500 "$PY" scripts/cache_lewm_latents.py --checkpoint "$ck" \
     --data-root "$RR" --render-root "$RT" --allow-material-color-render --seq-len 11 --stride 5 \
     --holdout-role eval --max-windows 512 --include-pose-labels \
     --device cpu --output "$evc" >>"$log" 2>&1 || true

  # ---- PRIOR (no-regression) ----
  [ -f "$dec" ] || timeout 2700 "$PY" scripts/probe_lewm_rollout_horizons.py --checkpoint "$ck" \
     --data-root "$RR" --render-root "$RT" --allow-material-color-render \
     --horizons 1,2,3,5,8,10 --device cpu --output "$dec" >>"$log" 2>&1 || true
  [ -f "$mpc" ] || timeout 2700 "$PY" scripts/probe_lewm_receding_mpc_proxy.py --checkpoint "$ck" \
     --data-root "$RR" --render-root "$RT" --allow-material-color-render \
     --horizons 1,2,3,5 --replan-steps 4 --batch-size 16 --max-batches 16 \
     --device cpu --output "$mpc" >>"$log" 2>&1 || true
  [ -f "$nL2" ] || timeout 1800 "$PY" scripts/benchmark_lewm_closed_loop_mpc.py --checkpoint "$ck" \
     --task visible-beacon --mode kinematic --family open_obstacle_field --split test_id \
     --scene-limit 8 --policies lewm,bearing,hold,random --horizon 3 \
     --model-device cpu --backend cpu --output "$nL2" >>"$log" 2>&1 || true
  if [ ! -f "$nH" ]; then
    [ -f "$trc" ] || timeout 2700 "$PY" scripts/cache_lewm_latents.py --checkpoint "$ck" \
       --data-root "$RR" --render-root "$RT" --allow-material-color-render --seq-len 11 --stride 5 \
       --holdout-role train --max-sessions 200 --max-windows 2048 --device cpu --output "$trc" >>"$log" 2>&1 || true
    if [ -f "$trc" ] && [ -f "$evc" ]; then
      [ -f "$ehead" ] || timeout 1200 "$PY" scripts/train_lewm_energy_head.py --checkpoint "$ck" \
         --train-cache "$trc" --eval-cache "$evc" --horizons 3,5,8,10 --epochs 30 \
         --device cpu --output "$ehead" >>"$log" 2>&1 || true
      [ -f "$ehead" ] && timeout 1800 "$PY" scripts/benchmark_lewm_closed_loop_mpc.py --checkpoint "$ck" \
         --head-ckpt "$ehead" --task visible-beacon --mode kinematic --family open_obstacle_field \
         --split test_id --scene-limit 8 --policies lewm,bearing,hold,random --horizon 3 \
         --model-device cpu --backend cpu --output "$nH" >>"$log" 2>&1 || true
    fi
  fi

  # ---- GEOMETRY (the new objective) ----
  [ -f "$pg" ] || timeout 600 "$PY" scripts/probe_pose_geometry.py \
     --pose-head-ckpt "$posehead" --checkpoint "$ck" --eval-cache "$evc" \
     --pose-label-source actual --device cpu --output "$pg" >>"$log" 2>&1 || true
  [ -f "$nPoseCost" ] || timeout 2700 "$PY" scripts/diagnose_nav_cost.py \
     --checkpoint "$ck" --pose-head-ckpt "$posehead" --split test_id \
     --family open_obstacle_field --scene-limit 8 --horizon 3 --device cpu \
     --backend cpu --output "$nPoseCost" >>"$log" 2>&1 || true
  [ -f "$nPose" ] || timeout 1800 "$PY" scripts/benchmark_lewm_closed_loop_mpc.py --checkpoint "$ck" \
     --pose-head-ckpt "$posehead" --task visible-beacon --mode kinematic --family open_obstacle_field \
     --split test_id --scene-limit 8 --policies lewm_pose,bearing,hold,random --horizon 3 \
     --model-device cpu --backend cpu --output "$nPose" >>"$log" 2>&1 || true

  # Never mark a checkpoint complete after a timeout, truncated JSON, or failed
  # optional-head training. Leave caches in place so the next pass can retry.
  if ! "$PY" - "$dec" "$mpc" "$nL2" "$nH" "$pg" "$nPoseCost" "$nPose" <<'PYEOF' >>"$log" 2>&1
import json, pathlib, sys
missing = []
for raw in sys.argv[1:]:
    path = pathlib.Path(raw)
    try:
        with path.open() as fh:
            json.load(fh)
    except Exception as exc:
        missing.append(f"{path}: {exc}")
if missing:
    print("incomplete eval outputs; retrying on next watcher pass:")
    print("\n".join(missing))
    raise SystemExit(1)
PYEOF
  then
    return
  fi

  # ---- consolidate one report row ----
  if ! "$PY" - "$ep" "$dec" "$mpc" "$nL2" "$nH" "$pg" "$nPoseCost" "$nPose" "$REPORT" <<'PYEOF' >>"$log" 2>&1
import json, math, sys
ep, dec, mpc, nL2, nH, pg, nPoseCost, nPose, report = sys.argv[1:10]
def L(p):
    try: return json.load(open(p))
    except Exception: return None
def navprog(p, pol):
    d=L(p)
    try: return d["aggregate"][pol]["mean_progress_m"]
    except Exception: return None
def fmt(x, f=".3f"):
    return ("{:"+f+"}").format(x) if isinstance(x,(int,float)) else "—"
# zero-free@10 and free/pers@10 from decomp (point metrics at horizon 10)
zf=fp=None; d=L(dec)
if d:
    for h in d.get("horizons", []):
        if h.get("horizon")==10:
            pm=h.get("point_mse",{})
            if "zero_action" in pm and "rollout" in pm: zf=pm["zero_action"]-pm["rollout"]
            fp=h.get("point_rollout_over_persistence")
m=L(mpc); mvs0=None
if m:
    for h in m.get("horizons", []):
        if h.get("horizon")==5: mvs0=h.get("recorded_win_rate_vs_zero")
g=L(pg)
ge=g.get("encoded_to_encoded") if g else None
gp=g.get("predicted_to_encoded") if g else None
pc=L(nPoseCost)
pc=pc.get("aggregate",{}).get("pose_single") if pc else None
required = (
    zf, fp, mvs0, navprog(nL2,'lewm'), navprog(nH,'lewm'),
    ge.get('pose_xy_err_m') if ge else None,
    ge.get('dist_pearson') if ge else None,
    gp.get('pose_xy_err_m') if gp else None,
    gp.get('dist_pearson') if gp else None,
    pc.get('mean_first_spearman') if pc else None,
    pc.get('mean_first_regret_m') if pc else None,
    navprog(nPose,'lewm_pose'),
)
if any(value is None or not math.isfinite(float(value)) for value in required):
    raise RuntimeError("one or more required report metrics are absent or non-finite")
row=f"| e{ep} | {fmt(zf)} | {fmt(fp)} | {fmt(mvs0,'.2f')} | {fmt(navprog(nL2,'lewm'))} | "\
    f"{fmt(navprog(nH,'lewm'))} | **{fmt(ge.get('pose_xy_err_m') if ge else None)}** | "\
    f"**{fmt(ge.get('dist_pearson') if ge else None,'+.3f')}** | "\
    f"**{fmt(gp.get('pose_xy_err_m') if gp else None)}** | "\
    f"**{fmt(gp.get('dist_pearson') if gp else None,'+.3f')}** | "\
    f"**{fmt(pc.get('mean_first_spearman') if pc else None,'+.3f')}** | "\
    f"**{fmt(pc.get('mean_first_regret_m') if pc else None)}** | "\
    f"**{fmt(navprog(nPose,'lewm_pose'))}** |"
open(report,"a").write(row+"\n")
print("ROW:", row)
PYEOF
  then
    echo "report consolidation failed; retrying on next watcher pass" >> "$log"
    return
  fi
  rm -f "$trc" "$evc" 2>/dev/null || true
  touch "$flag"
  echo "=== $(date) done $name ===" >> "$log"
}

# Exit only after a guaranteed final sweep. While training cells are alive the
# loop keeps polling; once LADDER_SENTINEL disappears (driver finished/crashed) or
# — when no sentinel is configured — no pose-aux training is left, we do one more
# full sweep so the last epoch checkpoint of every cell is always evaluated.
final=0
WLOG="$ROOT/pose_aux_watcher.log"
while true; do
  for DIR in $DIRS; do
    mkdir -p "$DIR"
    REPORT="$DIR/finetune_eval_report.md"
    ensure_report_header "$REPORT"
    for ck in $(ls "$DIR"/lewm_seq11_e*.pt 2>/dev/null | sort -V); do
      eval_one "$ck"
    done
  done
  [ "$final" = 1 ] && { echo "final sweep done $(date); watcher exit" >> "$WLOG"; break; }
  if [ -n "${LADDER_SENTINEL:-}" ]; then
    [ -f "$LADDER_SENTINEL" ] || { echo "sentinel gone $(date); one final sweep" >> "$WLOG"; final=1; continue; }
  else
    pgrep -f 'train_lewm.py.*pose-aux' >/dev/null 2>&1 || { echo "training ended $(date); one final sweep" >> "$WLOG"; final=1; continue; }
  fi
  sleep 600
done
