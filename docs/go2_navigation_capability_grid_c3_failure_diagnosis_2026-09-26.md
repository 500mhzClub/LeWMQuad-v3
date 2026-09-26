# C3 development-screen failure diagnosis

The corrected C1 screen produced three round trips, four timeouts and three
pose losses. Timeouts are the largest outcome group. All ten missions had zero
disallowed contacts, hard-clearance violations or operating-margin violations.
The original five planning episodes passed exact containment. These are
development results, not validation capability estimates.

This analysis uses the completed `v0_grid_c3_C1_screen` logs. Sensor regeneration
uses only the original command tapes and unchanged tracker. No new navigation
trial, model fitting, controller change or sealed-set inspection contributes to
the diagnosis. The association between startup mode and failure is exploratory:
the maze, starting position and heading differ between episodes.

## Timeouts

| Episode | Beacon verified | Remaining shortest path at 480 s | Selected holds | Executed zero commands | Diagnosis |
|---|---|---|---|---|---|
| 05/0 | Yes, 103.6 s | 6.411 m to home | 586/1,194 (49.1%); return 61.4% | 65.1% | Prolonged visual-recovery deadlock on return, then late progress |
| 06/0 | No | 5.986 m to beacon | 70/380 (18.4%) | 72.6% | Failed route acquisition and exhausted view search; intermittent recovery turns without meaningful translation |
| 08/0 | No | 10.681 m to beacon | 0/74 (0%) | 93.8% | Exhausted view search; nearly stationary for the rest of the mission |
| 09/0 | No | Unresolved under the declared inflated reference | 971/1,100 (88.3%) | 90.1% | Visual recovery blocked by observed-memory clearance, followed by sustained hold |

Distances use the preregistered true-wall reference: 0.46-m disk inflation,
0.005-m additional clearance, 0.02-m grid. Episode 09 ends with base-centre
wall separation 0.4570 m, 8.0 mm inside that reference's required separation.
Its reference distance is therefore unavailable, rather than an invented or
unlabelled snapped distance. This does **not** contradict the articulated safety
result: the actual robot geometry and a 0.465-m circular reference differ.

The hold denominator is the preregistered number of scored selections. An early
planning return is not a selected hold. To prevent that denominator hiding
stalls, the accompanying tables separately report unscored planning calls and
the fraction of 20-ms intervals with an actually applied zero command.

**05/0.** The outbound leg succeeds. After beacon retrieval, the recovery target
repeatedly suppresses normal route following. At 150, 250 and 350 s, all six
candidates pass the recorded memory-clearance checks, but the view objective
admits only hold and turns. Heading errors are already approximately ±0.055 rad;
the hold utility exceeds both turning utilities. The recovery nevertheless
remains active. From 120–360 s, successive one-minute net displacements are
26, 1, 12 and 1 mm. These are sustained stalls, not a long navigational loop.
Movement resumes late: the final minute gains about 0.76 m in reference
distance, insufficient to recover the lost time. Of 586 holds, 572 are
movement-outscored cases, 12 have no eligible movement and two are arrival
overrides. Observation restrictions and motion exclusions are recorded
separately and can overlap.

**06/0.** View search first reports `VIEW_BUDGET_EXHAUSTED` at 29.6 s. There are
817 such planning returns across the mission. Other planning attempts resume
intermittently, but no observed-floor route to a frontier or goal is selected.
The total native base path is 3.876 m, largely local turn/gait movement; every
one-minute net displacement is below 0.10 m. The original 5.891-m beacon
distance ends at 5.986 m. Of 70 selected holds, 58 are the motion-dependent
observed-floor footprint override, nine have no eligible movement and three
lose on score. This is blocked route/view recovery, not slow useful progress.

**08/0.** The same view budget first expires at 29.6 s, producing 1,123 unscored
planning returns. Only 74 scored decisions occur in 480 s, all turns. Entire
minutes execute zero commands. Later occasional turn decisions do not establish
a route. The initial 10.669-m beacon distance ends at 10.681 m. This is a
stationary planning deadlock despite the misleading 0% selected-hold rate.

**09/0.** All 971 holds have no eligible movement. The view objective excludes
forward and both arcs; the unchanged memory-clearance rule excludes both turns.
After 60 s, all applied commands are zero. Later floor-reacquisition holds add
another dispatch/planning restriction. The robot has no meaningful progress
after its initial local movements; this is blocked recovery, not mission-scale
oscillation or slow traversal. No safety margin is relaxed by this diagnosis.

## Shared startup pattern

All five startup-fixed episodes use the existing auxiliary camera to initialise
the floor. They begin planning at 1.2 s facing a nearby wall with almost no
observed floor in the routing map. The other five begin with substantially more
floor evidence. The relevant startup association is therefore the floor/view
condition, not simply whether the earlier map bound was exceeded.

| Episode | Startup floor source | Floor cells at first plan | Primary optical centre ray to wall |
|---|---|---:|---:|
| 00/0 | Primary | 868 | 1.618 m |
| 01/0 | Auxiliary | 0 | 0.407 m |
| 02/0 | Auxiliary | 0 | 0.356 m |
| 03/0 | Primary | 681 | 1.586 m |
| 04/0 | Primary | 1,933 | 1.598 m |
| 05/0 | Primary | 551 | 1.636 m |
| 06/0 | Auxiliary | 6 | 0.285 m |
| 07/0 | Primary | 324 | 0.754 m |
| 08/0 | Auxiliary | 0 | 0.317 m |
| 09/0 | Auxiliary | 2 | 0.286 m |

The ray distances are evaluator-only intersections from the actual camera
origin and optical axis with the registered wall boxes. They are not sensor
inputs or articulated clearances. The 0/5 versus 3/5 split identifies a shared
condition worth addressing; it does not establish that auxiliary-camera use
itself causes failure. By 28 s, 06 has acquired 1,114 floor cells and 08 has
251, yet both still lack a usable route. Initial map sparsity is consequently
not the only mechanism.

## Pose-loss diagnosis

The three failures occur during local view-seeking/recovery turns near walls,
not high-speed translation. The existing tracker already attempts both cameras;
adding a nominal downward-camera fallback would duplicate an existing path.
Primary images are dominated by close, broad wall checker squares, while the
downward camera sees a narrow floor strip beneath the wall. The depth near
limit is 0.2 m; RGB remains visible inside that limit even when its depth packet
contains no usable primary pixels.

| Episode | Failure time | Last 10 s absolute yaw travel | Last 10 s minimum articulated clearance | Terminal primary optical wall range |
|---|---:|---:|---:|---:|
| 01/0 | 341.0 s | 0.839 rad | 0.149 m | 0.169 m |
| 02/0 | 58.5 s | 4.334 rad | 0.173 m | 0.186 m |
| 07/0 | 249.0 s | 3.848 rad | 0.170 m | 0.188 m |

01's last applied command is right turn; 02's is left turn; 07's is right turn.
The camera's proximity does not indicate a robot collision. All three retain
substantial articulated separation. A physically safe turn can still make the
visual pose unobservable.

01 rejects the auxiliary rigid fit on the combined consensus/grid-support/
displacement gate after the primary loses all valid depth. 02 rejects auxiliary
matches for insufficient noncollinear, well-spread support; its primary also
lacks sufficient rigid matches. These are the actual rejection messages; the
combined 01 message does not identify which individual subcondition failed.
07 likewise rejects the auxiliary rigid fit on the combined consensus/grid-
support/displacement gate. Primary valid-depth counts at the failure are
0/307,200 (01), 21,428/307,200 (02), and 69,823/307,200 (07); 07's immediately
preceding frame has zero. Last-accepted feature witnesses are not misreported
as current-frame features.

All three command replays passed: **6,488 consumed RGB-D pairs matched bitwise,
6,485 accepted raw poses matched exactly, and every native state matched**.
The original terminal rejection was reproduced in each case. Total replay
wall time was 963.0 s, within its one-hour bound.

## First change selected: qualified paired-plane floor initialisation

A bounded regeneration of the first 1.2 s before planning, with the unchanged
mapper and logged accepted registered poses, found a specific shared mapping
fault. All five auxiliary-start episodes fix the floor 23–29 cm above the
already-qualified paired depth plane. The control episode differs by less than
1 mm. These are sensor-derived plane comparisons; no true pose enters the fix.

| Episode | Old fixed map floor | Qualified paired-plane floor at initial body XY | Old minus paired |
|---|---:|---:|---:|
| 00/0 control | −0.320066 m | −0.319334 m | −0.000731 m |
| 01/0 | −0.085249 m | −0.319183 m | +0.233934 m |
| 02/0 | −0.079122 m | −0.319521 m | +0.240398 m |
| 06/0 | −0.029107 m | −0.319457 m | +0.290350 m |
| 08/0 | −0.059178 m | −0.319458 m | +0.260280 m |
| 09/0 | −0.032757 m | −0.319201 m | +0.286445 m |

The startup fallback accepted a median from auxiliary mesh-normal candidate
quads as the floor height. Those candidates do not establish the floor's
identity. The mapping code then retains that erroneous height throughout the
episode: floor coverage requires every relevant raw pixel to lie within 10 mm
of it, and obstacle-height marking uses it too. This explains the initial empty
floor maps and contaminates subsequent map interpretation. It provides a
concrete upstream mechanism affecting both timeout and pose-loss episodes;
whether it prevents their eventual failures requires the next screen.

**`v1_paired_floor` makes one change:** establish the map's initial scalar floor
height from the existing, qualified current paired-camera plane. It retains
the original quiet-gravity map orientation. If that plane is unavailable, use
the existing bounded startup acquisition/recovery path; do not accept the
unqualified median. Once established, the floor stays fixed as before.
The plane's acceptance criteria, subsequent mapping, tracker, models,
candidate bank, footprint margins and dispatch checks are unchanged. Every
controller receives the identical mapping change.

This is an outcome-driven mapping intervention and consumes one version:
**two of six used**, including the previously charged C2. No separate visual-
recovery or view-budget change is bundled into it. The return-recovery deadlock
in 05 and late pose loss in 07 are independently evidenced residual mechanisms;
they may remain after this change. Success is not assumed.

All six short mapping replays passed: 78 consumed RGB-D pairs matched bitwise,
native arrays matched exactly and first-plan retained floor/obstacle counts
reproduced their original logs. They used 16.2 simulated seconds including
settling, under a ten-minute wall cap. Results are retained in
`grid_c3_initial_map_diagnosis_attempt001`; no frames were retained.

## Evidence and interpretation limits

Detailed per-minute counts, input hashes, logged poses and reference queries:
`grid_c3_failure_log_diagnosis_attempt001/dev00.json` through `dev09.json`
under the capability RecoveryStorage output root. The analysis source is
`scripts/analyse_go2_capability_grid_c3_failures_readonly_development.py`.

The distance fields within each episode's time windows use its **terminal
leg's target** throughout. Thus the early 05 windows show distance to home,
even while outbound; they must not be interpreted as outbound progress.
Reported outbound completion comes from the existing physical arrival reader.
Exclusion counts overlap and are not causal effect estimates.

The pose replay root is `grid_c3_pose_loss_diagnosis_attempt001`. It preserves
each rejection record and verifies sensor hashes, native arrays and all logged
accepted raw poses. Inspection sheets are temporary RAM files removed after
viewing; the scalar diagnoses and their source identities remain retained.

The floor-initialisation fault is common to five failures, but is not a proven
single cause for all seven. No tracker acceptance threshold, safety rule,
model or candidate bank is changed. The next ten-episode C1 screen tests only
the paired-plane initialisation intervention, retaining every failure and
disqualifying the version on any disallowed contact or hard-criterion violation.

## One-minute timeout accounting

O = observation-only movement exclusions; M = predicted motion/memory-clearance exclusions; F = predicted footprint coverage override; A = arrival override. Counts overlap. “Outscored” is movement eligible but losing to hold. Blank scored denominators remain undefined. Zero-command percentages count all executed 20-ms intervals.

### 05/0

| Time (s) | Holds / scored | Unscored exits | Hold category counts | Exclusion / override counts | Executed zeros | Net displacement |
|---|---|---|---|---|---|---|
| 0–60 | 3/147 | native_dense_context_warmup 2 | outscored 2, ineligible 1 | M 3 | 4.3% | 1.453 m |
| 60–120 | 23/147 | mission_hold 3 | outscored 17, ineligible 4, override 2 | O 21, M 4, stopping 1, A 2 | 27.3% | 4.308 m |
| 120–180 | 103/150 | — | outscored 103 | O 103 | 91.2% | 0.026 m |
| 180–240 | 119/150 | — | outscored 119 | O 119 | 97.8% | 0.001 m |
| 240–300 | 127/150 | — | outscored 127 | O 127 | 97.3% | 0.012 m |
| 300–360 | 126/150 | — | outscored 126 | O 126 | 98.0% | 0.001 m |
| 360–420 | 66/150 | — | outscored 66 | O 66 | 65.0% | 0.048 m |
| 420–480 | 19/150 | — | outscored 12, ineligible 7 | O 19, M 7 | 39.5% | 0.732 m |

### 06/0

| Time (s) | Holds / scored | Unscored exits | Hold category counts | Exclusion / override counts | Executed zeros | Net displacement |
|---|---|---|---|---|---|---|
| 0–60 | 1/78 | native_dense_context_warmup 2, view_budget_exhausted 69 | outscored 1 | O 1 | 48.2% | 0.082 m |
| 60–120 | 19/94 | view_budget_exhausted 56 | outscored 1, ineligible 1, override 17 | O 19, M 18, F 17 | 45.7% | 0.075 m |
| 120–180 | 8/20 | view_budget_exhausted 130 | override 8 | O 8, M 8, F 8 | 91.2% | 0.025 m |
| 180–240 | 0/19 | view_budget_exhausted 131 | — | — | 86.0% | 0.039 m |
| 240–300 | 9/59 | view_budget_exhausted 91 | outscored 1, ineligible 2, override 6 | O 9, M 8, F 6 | 65.7% | 0.021 m |
| 300–360 | 33/86 | view_budget_exhausted 64 | ineligible 6, override 27 | O 33, M 33, F 27 | 61.5% | 0.095 m |
| 360–420 | 0/18 | view_budget_exhausted 132 | — | — | 86.7% | 0.026 m |
| 420–480 | 0/6 | view_budget_exhausted 144 | — | — | 95.7% | 0.022 m |

### 08/0

| Time (s) | Holds / scored | Unscored exits | Hold category counts | Exclusion / override counts | Executed zeros | Net displacement |
|---|---|---|---|---|---|---|
| 0–60 | 0/72 | native_dense_context_warmup 2, view_budget_exhausted 75 | — | — | 51.7% | 0.066 m |
| 60–120 | 0/0 | view_budget_exhausted 150 | — | — | 100.0% | 0.017 m |
| 120–180 | 0/0 | view_budget_exhausted 150 | — | — | 100.0% | 0.004 m |
| 180–240 | 0/1 | view_budget_exhausted 149 | — | — | 99.2% | 0.002 m |
| 240–300 | 0/0 | view_budget_exhausted 150 | — | — | 100.0% | 0.003 m |
| 300–360 | 0/0 | view_budget_exhausted 150 | — | — | 100.0% | 0.000 m |
| 360–420 | 0/1 | view_budget_exhausted 149 | — | — | 99.2% | 0.008 m |
| 420–480 | 0/0 | view_budget_exhausted 150 | — | — | 100.0% | 0.008 m |

### 09/0

| Time (s) | Holds / scored | Unscored exits | Hold category counts | Exclusion / override counts | Executed zeros | Net displacement |
|---|---|---|---|---|---|---|
| 0–60 | 18/147 | native_dense_context_warmup 2 | ineligible 18 | O 18, M 18 | 20.5% | 0.205 m |
| 60–120 | 150/150 | — | ineligible 150 | O 150, M 150 | 100.0% | 0.006 m |
| 120–180 | 150/150 | — | ineligible 150 | O 150, M 150 | 100.0% | 0.000 m |
| 180–240 | 150/150 | — | ineligible 150 | O 150, M 150 | 100.0% | 0.004 m |
| 240–300 | 144/144 | floor_reacquisition_hold 4 | ineligible 144 | O 144, M 144 | 100.0% | 0.001 m |
| 300–360 | 108/108 | floor_reacquisition_hold 29 | ineligible 108 | O 108, M 108 | 100.0% | 0.009 m |
| 360–420 | 111/111 | floor_reacquisition_hold 30 | ineligible 111 | O 111, M 111 | 100.0% | 0.013 m |
| 420–480 | 140/140 | floor_reacquisition_hold 6 | ineligible 140 | O 140, M 140 | 100.0% | 0.001 m |
