# Fresh independent-layout RGB/body prediction collection V1

## Population and scope frozen before new physics

Use the existing structurally audited12-layout inventory without modification:
6 train,3 selection,3 development-evaluation layouts,120 episodes each,1,440
total. Every layout has5 contexts x2 histories x2 supports x6 action-duration
cells. Preserve exact episode order, geometry, appearance/physics seeds, initial
states, native support conditions, command vocabulary and target times. Layout
roles are not final/sealed evaluation and no predecessor episode is eligible.

This is prediction-data acquisition, not a learned policy or navigation assay.
Only declared RGB/body/control tensors and target-side native motion/contact
labels may reach the later frozen matched study. Depth, shadow tracking,
privileged pose/map/friction metadata and final task success are not model
inputs. No model fitting is performed by this collection or audit.

## Fixed acquisition and per-modality eligibility

The collector directly reuses the completed core-ordered pilot's `collect`
function:fixed CPU learned gait, union visual surfaces, physical collision boxes,
floor-first draw ordering, core-profile framebuffer query, rigid mounted RGB-D,
causal ideal body sensors, exact requested commands and native physical stops.
The pilot's four-complete-repeat criterion and strict depth failures remain
unchanged. Its runs and the failed earlier independent batch are not resumed,
rescored for eligibility, imported as new cases or used for training.

Every committed case undergoes full raw sensor/contact/command/setup/first-stop
reconstruction, terminal-event coverage and raster order/precision verification.
A complete schedule or completely recorded physical terminal after departure,
with complete causal history and faithful targets, is eligible for the RGB/body
prediction dataset. Setup failures, pre-departure stops, infrastructure truncation
and corrupt recordings remain explicit in the planned denominator, not favorable
negative contact labels. Cumulative contact can be known beyond a collision stop;
future motion and images after the stop remain censored. No zero-motion padding.

Keep original strict1mm depth scores and measurement-independent pixel-footprint
partitions. Stable-interior errors or near-plane/false-valid occlusion failures
are hard measurement failures:write the complete precheck, stop the batch and
preserve all evidence. Boundary-only strict failures remain reported failures,
but are not a sole exclusion criterion for RGB/body prediction when raw/capture/
terminal evidence otherwise passes. They grant no depth-navigation qualification,
pixel repair, free-space certificate or hardware-sensor validity.

All same-context action prefixes are compared to action0, including unequal or
unavailable pairs. A pair's counterfactual validity is separate from individual
supervised data coverage. No successful-subset or tracking-based filtering.

## Execution, source and resource bounds

Use distinct source paths and exclusive external children
`go2_independent_rgb_body_collection_v1_l00_attempt_001` through
`go2_independent_rgb_body_collection_v1_l11_attempt_001` of the existing
development artifact root. No recursive source export or artifact discovery.
Bind the exact source closure, old native bindings, inventory and completed
coverage/native-edge-assay witnesses in each serialized launch before execution.
One batch at a time; before l01 onward, the previous fixed batch must have a
complete collection and120-case raw audit with no hard measurement failure.
No retry/resume, case substitution or role reassignment under V1.

`run_go2_independent_rgb_body_stage_v1.py --batch l00` orchestrates one fresh
batch followed by its terminal audit, including when collection produces a
terminal failure. It rejects any pre-existing output before collection and does
not audit a preflight failure without a terminal record. It never restarts a
batch and does not launch the next layout automatically.

Per episode:max2,400 physics samples,34 RGB-D frames and33 command ticks.
Per batch:8GiB total artifact budget,256MiB episode allowance and40GiB free
reserve. Serialized collection launch is capped at32MiB. The terminal audit
has a128MiB metadata cap and shares the8GiB batch total. Commits bind an exact
explicit union-scene/raster/sensor/native/metadata roster, absent artifacts and
actual byte counts; raw prechecks are durable before advancing. Infrastructure,
storage, integrity and hard measurement failures terminate the batch without
deleting partial evidence. Physical stops and failed setups remain population
outcomes and do not trigger replacement cases.

The terminal auditor requires exactly one collection terminal record, recomputes
all committed raw evidence and matches saved prechecks. It retains all departure
windows/targets separately from eligible RGB/body windows/targets, materializes
eligible samples through the existing dataset interface, preserves inventory
roles and reports all120 case statuses. No fitting until population coverage is
known and the matched learning study is frozen.

## Scientific exit conditions, not goal completion

Completion of the intended acquisition stage requires all12 fixed batches with
honest missingness, physical failures, modality eligibility, action coverage and
pairing evidence. The ability to fit all matched arms still depends on actual
eligible train/selection/evaluation coverage; do not assume it in advance.
Then compare direct/supervised-rollout/JEPA arms under shared exposure and3 paired
seeds, action/time and zero-motion baselines and RGB/history/action ablations.

Reliable local stopping/turning, actual online rollout and memory/backtracking
benefit, novel-maze completion, runtime depth uncertainty/thin-obstacle handling,
real-time deadlines, self-occlusion and realistic camera/body sensing, and bounded
supervised hardware evidence remain required for the full goal. Hidden-robot
rendering and ideal sensor noise/latency remain simulation limitations.
