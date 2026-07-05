# Go2 JEPA Geometric-Encoder Iteration — 2026-07-02

Continues `lewm_go2_fully_learned_tier_execution_2026-07-01.md`. User directive:
iterate until fully-learned held-out nav, JEPA-based (no DINO). This report
covers the representation rebuild, the safety result (achieved), and the
coverage wall (observability-limited — decision needed).

Suite: `.generated/go2_memory_closed_loop/generalized_learned_local_suite_v217_replacement_cap_20260701/`
Dense data: `~/go2_head_dagger_dense/` (61 scenes × ~800 poses).

## Representation line (all JEPA-architecture, geometric aux supervision)

Offline gate = clearance-head accuracy on 2 held-out-scene closed-loop rows.

| encoder | data | gate acc | notes |
|---|---|---|---|
| frozen contrastive (baseline) | — | 0.581 | chance at margin in-loop |
| frozen, spatial tap (arm 1) | — | 0.584/0.580 | geometry not in trunk — FAILED |
| geo v1 (init from frozen) | 11 medium scenes dense | 0.764 | +0.18: supervision works |
| geo v2 (lat192, scratch) | + 18 cross-family scenes | 0.775 | underfit at 64px |
| geo v3 (img128) | same | 0.799 | forward AUC 0.856 |
| geo v4 (img128) | + 30 more medium (41 total) | 0.811 | **flat — data no longer the lever** |

Outcome-head labels: swept-min clearance label was 92%-positive and saturated
(v224) → reranker erased forward proposals. Progress-floor label (v227) was
collision-blind (sweep integrated through walls) → wall-ramming. Fixed with
collision-aware counterfactual progress (v230/v235: sweep stops accruing at
first swept contact) — operationally exact for the kinematic sim.

## Closed-loop line (held-out `eeb8320a6934`, seed 1, 700 ticks)

| run | stalls (hard) | forwards | cells | claimed |
|---|---|---|---|---|
| v218 old heads | 62 (44) | 146 | 16 | red |
| v225 geo-v3 heads | **0 (0)** | 55 | 14 | red |
| v231 + collision-progress outcome | **0 (0)** | 173 | 13 | red |
| v233 + SEEK/SERVO guard from 0 claims | **0 (0)** | 151 | 10 | red |
| v236 v4 heads thr082 | **0 (0)** | 169 | 6 | red |
| v237 progress-only guard | **0 (0)** | 122 | 7 | red |

Train probes match: 010092 94→2 stalls; 000c67 SERVO-hammer 392→4 (guard in
SEEK/SERVO from zero claims — `--wall-guard-post-claim-min-claims 0` — without
breaking claims); 01732a clean with red.

Two additional structural fixes this session (committed):

- **Online-map mark-erasure bug** (`026447f`): a partial in-cell forward after
  a backward escape was credited with reaching the ahead cell and erased the
  stall mark — one wall edge pushed 214× (428 mark/unmark events). Success
  bookkeeping now requires a real cell transition. 01732a: 214→12 stalls,
  coverage 7→25 cells.
- **Native-res head inputs** (`8176040`): image_size≠64 heads were fed
  upsampled ego64.

**Safety on this stack is solved**: multiple zero-stall/zero-violation runs on
held-out and train scenes with 5-10× the prior translation volume.

## The coverage wall (blocker — decision needed)

No run sees more than red in 700 ticks. Root cause chain, each step verified:

1. Yaw-walk was guard-conversion → fixed by discriminative heads. Then:
2. **Privileged DFS explorer, same guard stack: 9 cells, 118 clean forwards**
   (`heldout_eeb832_vDIAG2_dfs_v4heads_seed1`). Not a policy problem.
3. Replay video (`heldout_v237_replay.mp4`): the robot's ego camera spends
   most ticks staring into a featureless wall at point-blank range in
   body-width corridors. Frames carry no information about where passages are.
4. Ground truth: **36-44% of free space in this maze family has obstacle
   clearance ≤ 0.24 m — the robot's half-width** (held-out 0.36, train scenes
   0.35-0.44). Passages leave 2-5 cm margins.

Conclusion: with a 0.48 m-wide body, egocentric-RGB-only frame-wise decisions,
and a ZERO contact-like-stall gate, threading 2-5 cm-margin gaps sight-unseen
is at/near the observability limit — the perception AUC plateau (~0.85-0.86
across v3/v4 despite 4× data) is the signature. The earlier scaffolded demo
threaded this geometry with exact grid routes.

## Options (user decision)

A. **Contact-tolerant gate**: allow a small budget of low-speed contact-like
   stalls (e.g. `--max-contact-like-stalls 8-15`, zero hard-collision/body
   violations). The fixed online map makes each wall touch a one-time,
   recovered event — this is how animals/insects explore dark tight spaces.
   Likely reachable with current components + coverage tuning.
B. **Ego-depth contract**: depth occupancy already fully completed a maze
   (v43 line). Perception blocker disappears; the learned-memory/policy claims
   stand on depth-based occupancy instead of RGB-only.
C. **Wider demo family**: run the identical fully-learned stack on a family
   whose passages exceed body width comfortably (e.g. open_obstacle_field or
   a wider maze generator config). RGB-only zero-contact is plausible there.
D. **Belief accumulation over edges** (multi-view averaging of head
   predictions per (cell,yaw)): in-contract, but the video suggests little to
   accumulate — wall-filled frames are uninformative from every nearby pose.
   Highest effort, lowest expected payoff of the four.

Recommended: A (keeps the RGB-only fully-learned claim, honest about contact
as sensing) or C (keeps zero-contact, changes the arena), in that order.

## Assets added this session

- `scripts/train_go2_jepa_geometric_encoder.py` — geometric-aux JEPA retrain,
  checkpoint drop-in for `load_go2_jepa_encoder`.
- `scripts/build_go2_head_dagger_frames_from_result.py --sample-poses` —
  dense free-space pose frames from any scene.
- Trainer: `--feature-mode spatial`, `--counterfactual-blocked-source progress`
  (collision-aware), `--image-size` per-head plumbing.
- Benchmark: per-head encoder overrides, native-res head inputs, online-map
  mark-erasure fix.
- Checkpoints: `go2_jepa_geometric_encoder_v{1..4}*.pt`, heads v221-v235.
