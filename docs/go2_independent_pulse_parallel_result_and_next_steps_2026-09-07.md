# Independent-layout learning result and next steps — 2026-09-07

## Outcome

The 12-layout collection and the complete 36-fit matched learning study finished.
The present experiment does **not establish an RGB-JEPA advantage** over the
action/time baseline, and it does not train or evaluate a closed-loop maze policy.
The added latent objective has a small, conditional position-prediction benefit
against supervised rollout, alongside negative comparisons and an insufficiently
vision-dependent action-ranking task. Preserve all of these results.

The full objective remains sensor-grounded Go2 novel-maze navigation with useful
JEPA prediction, online memory, controlled causal comparisons, realistic timing
and bounded hardware evidence. No BEV direction is introduced. This report
supersedes earlier *live/preflight/not-launched* status descriptions for the
learning study only; frozen protocols and historical reports remain unchanged.

## Completed evidence and limits

Collection: 12 layouts, 1,440 eligible trials, comprising 1,380 scheduled completions
and 60 physical-terminal recordings. There were no setup, pre-departure,
infrastructure-truncated, invalid or unattempted trials. The 256 positive-contact
targets are horizon labels, not 256 independent episodes or successful missions.
See the [collection result](go2_independent_rgb_body_collection_result_2026-09-07.md).

Learning: three seeds × four information treatments × three objectives, 1,200
updates per fit, batch six: 36 fits, 43,200 optimizer updates and 259,200 draws.
Final checkpoints only; identical per-seed initialization and draw schedules.
Equal update/data budgets are not equal FLOPs or active parameter counts.
Six training, three selection and three development-evaluation layouts were
fixed in advance. Seeds are crossed repeats on the same layouts: evaluation has
**three layout units, not nine independent environments**.

All fits and the parent terminated normally. The terminal reports
MATCHED_DEVELOPMENT_COMPARISON_COMPLETE. Its elapsed interval was 5,899.49 seconds
(98.32 minutes), excluding initial preflight; the command took about 107 minutes
including preflight. Preterminal artifacts total 334,152,680 bytes. Four CPU
workers ran concurrently, sampled at approximately one fully utilized CPU each,
with about 1.7 GiB peak RSS per process. This used four cores, not the entire
16-core machine; GPU fitting was not part of the frozen numerical design.
The earlier synthetic speedup is not a measured real-cohort serial speedup.

The completed reader authenticated the 416 preterminal artifact bindings,
36 jobs and fit receipts, schedules, optimizer ledgers, source/configuration
identity and original collection receipt, then aggregated the saved scores.
This is **not independent recomputation of training or raw prediction scoring**.
A subsequent exact-path check reconfirmed the terminal, coverage and all three
development score identities. No checkpoints or raw images were loaded for this
readout.

## Complete primary-model comparison

Development-evaluation layout/seed macro means, lower is better. Motion metrics
use only available motion targets; contact metrics have a larger denominator.
These are descriptive development results, not final evaluation or calibrated
deployment uncertainty. The direct objective uses its direct head; supervised
rollout and JEPA use their recursive outcome heads.

| Primary model / input treatment | Position error (mm) | Yaw error (mrad) | Contact Brier |
| --- | ---: | ---: | ---: |
| full_jepa_rollout_outcomes | 22.828 | 55.488 | 0.032165 |
| full_supervised_rollout_rollout_outcomes | 25.834 | 57.793 | 0.031539 |
| full_direct_direct_outcomes | 21.625 | 17.557 | 0.032511 |
| no_rgb_jepa_rollout_outcomes | 18.620 | 47.220 | 0.031644 |
| no_rgb_supervised_rollout_rollout_outcomes | 18.959 | 27.414 | 0.032411 |
| no_rgb_direct_direct_outcomes | 23.217 | 14.531 | 0.031621 |
| latest_packet_only_jepa_rollout_outcomes | 22.223 | 46.877 | 0.031681 |
| latest_packet_only_supervised_rollout_rollout_outcomes | 12.891 | 28.599 | 0.032234 |
| latest_packet_only_direct_direct_outcomes | 22.715 | 33.390 | 0.032167 |
| no_candidate_command_jepa_rollout_outcomes | 43.748 | 116.486 | 0.036274 |
| no_candidate_command_supervised_rollout_rollout_outcomes | 25.462 | 115.055 | 0.034762 |
| no_candidate_command_direct_direct_outcomes | 28.228 | 117.500 | 0.034661 |
| action_time | 18.223 | 9.052 | 0.031722 |
| zero_motion_empirical_contact | 26.115 | 110.554 | 0.031722 |

The [compact readout](go2_independent_pulse_parallel_scientific_readout_2026-09-07.json)
retains all 27 predeclared primary contrasts, all three roles and all seven
metrics, with missingness and cell counts. Values are rounded to twelve
significant digits there; bound original scores retain full precision and
horizon/stratum detail. The [secondary details](go2_independent_pulse_parallel_scientific_details_2026-09-07.json)
retain per-seed/per-layout motion and contact metrics, target availability and
the action-ranking diagnostic. No primary heads were missing. Negative oriented
deltas favor the left arm; concordance uses the reversed raw-score direction.

Principal interpretations:

- Full JEPA versus action/time: position error is 4.604 mm worse and yaw error
  46.436 mrad worse. Position improves in 3/9 paired cells and worsens in 6/9;
  yaw and both reported Brier comparisons worsen in all nine. There is no
  overall advantage on these outcomes.
- Adding the latent objective to full supervised rollout reduces position
  error by 3.007 mm (8/9 cells improve). Yaw improves by 2.305 mrad in the
  macro, but only 5/9 cells improve. Prediction Brier worsens by 0.000625 and
  matched two-second Brier by 0.001045. For the latter, 8/9 cells improve but
  one worsening dominates: aggregate direction is not uniform behavior.
- Full supervised rollout is worse than full direct prediction by 4.210 mm
  and 40.236 mrad; both worsen in all nine cells. Full JEPA versus direct
  combines the recursive package and latent objective, so it does not isolate
  either contribution.
- Removing RGB from JEPA improves macro position by 4.207 mm and yaw by
  8.269 mrad; motion improvement occurs in 6/9 cells and both Briers improve
  in all nine. This is evidence against benefit from RGB in this fitted
  configuration, not proof that RGB cannot help navigation.
- Removing the candidate command substantially harms JEPA: full input improves
  position by 20.920 mm and yaw by 60.997 mrad, in all nine cells. Action
  conditioning matters, but the action/time prior already solves the measured
  collision ranking.
- Latest-packet-only supervised rollout has the lowest descriptive position
  macro (12.891 mm), but its yaw remains worse than action/time. Do not promote
  it by selecting the best exposed development ablation. Latest-packet-only
  retains internal body/control histories, so this is not a no-memory test.
  No-candidate retains time/duration/past controls, so it is not action-free.

Seed variability is material: full-JEPA position means across the three layouts
are 31.193, 16.224 and 21.065 mm for the three seeds. Full-JEPA training and
development position macros are approximately 22.838 and 22.828 mm; action/time
is likewise similar across roles. A reasonable diagnostic priority is task
informativeness and model/optimization behavior, rather than assuming that
unseen-layout overfitting is the cause. These observations do not prove a
specific optimization failure, convergence or lack of geometric diversity.
No p-values or general reliability interval are inferred from three layouts.

## Why the task does not yet test the intended visual capability

| Role | Contact-labeled horizons | Motion + future-image targets | Positive-contact horizons | Matched action groups | Contact-contrasting groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 3,600 | 3,472 | 128 | 120 | 18 |
| Selection | 1,800 | 1,737 | 63 | 60 | 9 |
| Development evaluation | 1,800 | 1,735 | 65 | 60 | 9 |

All matched groups were scored at the common two-second horizon with the exact
same within-group sensor prefix. Every contact-contrasting group is near-wall.
The remaining 51/60 evaluation groups contain no contact contrast.

For all three seeds and every evaluation group, action/time's minimum-risk set
is the same four action IDs: **2, 3, 4, 5**. Its tie-averaged observed contact
fraction is zero. Every action-aware learned arm, action/time and the
zero-motion-with-empirical-contact baseline achieves zero minimum-score contact
fraction and zero avoidable-contact regret, with concordance 1 on the nine
contrastive groups. Concordance is unavailable on all-zero-contact groups;
1 is not evidence of 60 independent hard decisions. The no-candidate variants
perform worse, but this does not demonstrate visual use.

These are scored hypothetical selections from recorded candidates, not actions
executed by the evaluated model. A constant low-risk preference need not make
goal progress. The empirical-contact baseline is not a claim that all actions
have zero risk. More layout IDs alone cannot resolve this lack of a measured
need for scene-dependent decisions.

All 256 positive-contact horizons lack both a future image and a motion target.
Their contact-by-horizon label remains known after physical termination.
Consequently the latent future-image objective has no observed image target at
those positive-contact horizons, although earlier views and supervised contact
loss can still convey relevant information. This is a target-support limitation,
not permission to fabricate post-stop images, extrapolate unexecuted commands,
discard contact labels, or describe censored motion as observed.

## What worked, and what remains missing

The complete fixed cohort, prefix matching, causal input distinctions, retained
negative outcomes, three optimization seeds and full factorial enable more
defensible comparisons than a single favorable run. Deterministic parallel
fitting completed without changing scientific budgets. The all-arm result
makes a conditional latent-objective improvement visible without disguising
the stronger action-only baseline.

Still missing are a visual state/action interaction necessary for useful
progress, informative pre-contact predictive supervision, a successful online
rollout intervention, long-term memory benefit, sufficient independent layout
replication, calibrated uncertainty and deployment-valid sensing/timing.
The learned low-level gait and learned predictors do not turn the engineered
high-level executor into a learned navigation policy.

The latest actual continuous room-return result remains
[0/3](go2_inner_arrival_collection_result_2026-09-06.md): two tracking failures
during return turns and one low-friction excursion. Median observation/control
times there were 143–147 ms against a 100 ms interval with physics paused.
Neither this learning result nor successful recorded-stream pose availability
repairs those physical failures.

## Ordered next steps and decision gates

### 1. Run the already frozen independent tracking challenge

This is the immediate experimental priority and does **not** require a positive
JEPA result. The [frozen definition](go2_independent_tracking_challenge_v1_frozen_definition_2026-09-07.json)
and complete-result reader are ready; do not add another preparatory probe or
change frozen sources. Its eight trials are two scene clusters × two friction
conditions × two turn directions, not eight independent mazes. Fixed motion
tapes and paired replay test tracking, not closed-loop navigation.

Use only the frozen outside supervisor, with the actual complete study result
identity below. Authenticate all eight recordings, measured motion coverage,
base/stress comparisons and the outside terminal before interpreting the result.
Retain every failure; no retry, shortened tape, changed seed or direct unbounded
parent launch. A completed challenge would still require physical closed-loop
integration and mission evaluation.

Current exact-path preflight: both new output roots remain absent.
Available artifact-volume space was 94,555,316,224 bytes (about 88.06 GiB);
required space is 98,784,247,808 bytes (92 GiB), including the 40 GiB reserve.
Do not weaken this gate. The current blocker is approximately 3.94 GiB of
additional free space, not RAM or an ongoing training job. The old 23 GiB
TinyQuadJEPA environment is an identified candidate, but explicit deletion
approval remains unanswered. Nothing has been deleted. Recheck exact targets,
dependencies, processes and hardware after approval; moving files on the same
filesystem would not free space.

If tracking fails, localize actual error/coverage and use the existing
[conditional heading follow-up](go2_independent_heading_followup_plan_2026-09-07.md)
where warranted. Do not relax a rejection gate solely to pass rejected frames.
If it supports a candidate, test that candidate during actual uninterrupted
multi-leg control before claiming restored return capability. Low-friction
dynamics remain a separate problem even with accurate tracking.

### 2. Design the next learning test around necessary visual decisions

This is a prospective successor, not a reinterpretation or retry of this study.

- Construct balanced local situations where left/right or alternative progressing
  actions exchange their utility as visible geometry changes. Hold commanded
  actions, horizon and body/support history matched as closely as physically
  feasible; vary the relevant geometry independently of action IDs and appearance.
  Exact within-scene prefixes are achievable; do not falsely claim identical
  cross-scene physical histories. Audit differences and possible shortcuts.
- Require useful displacement toward a declared local goal as well as collision
  avoidance. Keep stopping available for safety, but do not score permanent
  stopping as navigation success. Before costly fitting, check on training-only
  coverage that one fixed action preference cannot solve the intended task.
  Retain failed coverage as a failed design, not a favorable subset.
- Collect genuine approach, interaction, turn, slip and braking transitions.
  Define actual-time pre-contact image/motion supervision and terminal-event
  masks prospectively. Report contact risk and observed latent/motion targets
  with their separate denominators. An explicit survival/contact objective can
  use known terminal labels without inventing missing future views.
- Split by independent geometry before fitting, and cross appearance separately.
  Audit geometry/start/prefix relationships rather than equating new IDs or
  one-pixel image differences with independence. Preserve a development split;
  future final evaluation remains external and custodian-isolated.
- Diagnose fit behavior using training/selection data: learning curves, per-action
  residuals, output/latent variability and dependence on observations. Preregister
  any new objective, architecture or budget before new evaluation; do not select
  the current best seed or ablation as a confirmed winner. Equalize observation,
  action, data and update budgets and additionally report compute/latency.

These requirements target the demonstrated shortcut and target-support gap.
They are not evidence that the proposed successor will succeed.

### 3. Connect reliable local execution to actual memory-guided missions

Follow the existing [execution-to-memory plan](go2_local_execution_to_online_memory_next_steps_2026-09-06.md):
reuse observed visit/attempt semantics and uninterrupted sensor frames rather
than creating another disconnected graph. Test a connected-maze departure,
branch choice, dead end/backtrack, hidden-marker observation and return home.
Unknown place association must remain possible in repeated-looking corridors.
Current observations or stored sensed observations select local targets; evaluator
pose and maze topology may score outcomes but must not choose online commands.

Compare persistent-memory and local-only agents with identical sensors, gait,
local execution, observation opportunities and mission budgets. Score actual
mission success, collisions, incorrect place/home claims, path/time cost,
revisits and all incomplete missions. A correct stack operation is not a physical
return. Freeze quantitative adoption criteria and an independent-layout
population before these runs; do not use component tests as acceptance evidence.

### 4. Separate prediction learning from online planning, then test deployment

With executable missions, compare direct reactive prediction, supervised
action-conditioned rollout and JEPA using matched resources. Separately switch
online multistep rollout on/off using the same frozen predictor and action/cost
interface, and switch persistent memory on/off. Measure whether predicted
differences actually change commands and improve completed missions. Prediction
error alone cannot establish the benefit of online planning or memory.

Then test the required sensor configuration under noise, dropout, self-visibility,
timing and synchronization limits; ideal RGB-D replay is not RGB-plus-real-sensor
validation. Run with physics advancing during computation and report deadline
misses and stop behavior. Reassess CPU/RAM/GPU/storage before each large job;
benchmark numerical compatibility and end-to-end throughput before selecting a
new device or concurrency for a future unfrozen experiment.

Bounded Go2 hardware work requires actual access, calibration and appropriate
physical supervision. It is not authorized merely by a simulation result.
The final goal requires the whole navigation, contribution, generalization,
timing and platform evidence chain; every stage here remains intermediate.

## Evidence identities

- Collection terminal: 5e39ff8b3b456578c13b878bdd12a030ffac62bd16cd257ca98f72e2e86e446d
- Study terminal: 588f24def6ec8810ae5a3411277576b0d965c77bf6ffdb8e18cfd80dce7b8122
- Study launch: 7af8abf6be59776cee4ea5970611d28fa9daf457b98d66ceb456a7dbfa40479e
- Study definition: 8d8c3456054a284aa83031ea417d8c433beddbcc04a8b47d3164f120bc0ae5d8
- Tracking definition: 223056ac7ddcb47b9d1a4b1b188028761fb15f56443f02fda28ecfa668326744
- Compact readout SHA-256: 8e66888abc6058d5291b601359def6a3c333bf4b361187ae3c1a5655f2484a27
- Secondary details SHA-256: 0131179b06661159ccc14e48ec49d80ed2f8cdb1ac49e1e0118dc57cb2bc7ca8

The attempt is the exact owned development root
/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_independent_pulse_parallel_study_v1_attempt_001.
No legacy runtime access, sealed discovery, source export, new training or native
execution was performed to prepare this report. All final-goal qualification
claims remain false.
