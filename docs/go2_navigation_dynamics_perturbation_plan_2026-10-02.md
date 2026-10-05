# Plan: dynamics perturbation (draft, not run) — 2 October 2026

**Status: next, straight after the new-harness re-run (Andrew, 3 October re-scope).** Option F of the [options note](go2_navigation_benchmark_discrimination_options_2026-10-02.md). It runs on the next harness version (reserve exit; [plan](go2_navigation_harness_reserve_exit_plan_2026-10-02.md)). **Order:** friction first, then low-friction patches with a visual marker. **Controllers:** C1, C3 and C4, recovery off. **Question:** does C3's or C4's visual prediction beat C1's command-only prediction when commands no longer determine motion? Moving obstacles (E2) wait until this is done.

## Question

When commands no longer determine motion, does vision-based prediction (C3 JEPA, C4 supervised) carry decision-relevant information that command history (C1) lacks?

In the preliminary run, C1's kinematic forecast was as accurate as C4's (6 mm) and drove as well. The forecast-sensitivity experiment ([results](go2_navigation_forecast_sensitivity_2026-10-02.md)) measures how much forecast error this harness tolerates before driving degrades. A perturbation only discriminates if it pushes C1's error past that point while the visual predictors stay inside it.

**Hypotheses:**
- **H1.** Under perturbed dynamics, C1's forecast error grows (it maps commands to nominal motion), while C3 and C4, which see recent frames, track the true motion more closely.
- **H2.** That error gap turns into a driving gap (success, SPL, time, contacts) once C1's error passes the harness's tolerance.
- **H0.** All three degrade alike: C3 and C4 learned the nominal command-to-motion mapping and use vision mainly for what they also get from commands.

## Perturbations

All are applied in the simulator only. The controllers, harness, sensors and evaluator are unchanged, and C0 stays a perfect-forecast reference because its oracle forecasts come from the perturbed physics.

| Perturbation | Mechanism (Genesis 0.4.6) | Effect on command → motion | Candidate levels |
|---|---|---|---|
| **Floor friction** | The maze spec's `friction_mu` (default 1.0) feeds `physics_randomization.floor_friction_mu`, the floor's Rigid material (clamped to 0.01–5). Earlier room-return experiments used 0.15 through the same path. | Foot slip: less translation and yaw rate than commanded, with more variance, and drift on turns. | μ = 0.6, 0.4, 0.25 |
| **Payload** | `robot.set_mass_shift(Δm, base link)`, plus optional `set_COM_shift` for an off-centre load. Go2 base link 6.92 kg, total 15.02 kg (Genesis `go2.urdf`). | Slower acceleration and lag; reduced speed on turns. | +2, +4, +6 kg (centred); +4 kg with COM shifted 5 cm |
| **Motor strength** | Scale the PD gains (`set_dofs_kp/kv`, nominal kp 20, kd 0.5 from the locomotion checkpoint) or the torque limits (`set_dofs_force_range`). | The locomotion policy tracks its joint targets less well: lower realised speed and sluggish turns. | kp × 0.7, 0.5; torque limit × 0.6 |
| **Mixed, within an episode** (second, after friction) | Low-friction floor patches with the existing `slick_patch` visual marker (`lewm_genesis/textures.py`). | Motion changes where the floor looks different: vision could anticipate it, command history cannot. | μ = 0.25 patches over 20–40% of route cells |

**Picking levels.** Before any navigation run, a short open-loop characterisation drives each candidate level with fixed command tapes on a flat floor. It measures:
- realised over commanded speed and yaw rate;
- their variance;
- whether the locomotion policy stays stable, with no falls.

The levels kept are those whose realised/commanded ratio falls where the forecast-sensitivity curve shows driving starts to degrade for a uniformly biased forecast, plus one beyond. Levels where the robot falls or cannot walk are dropped; they test locomotion, not prediction.

**Thresholds from the sensitivity results (added 2 October, evening; [results](go2_navigation_forecast_sensitivity_2026-10-02.md)).**
- **Friction, payload and weak motors make the robot move less than commanded.** C1 predicts nominal motion, so it over-predicts: its predicted/true ratio is commanded/realised.
  - On V4, over-prediction was tolerated to the limits tested: uniform × 1.5 gave 19/20 and turns × 2.0 gave 19/20.
  - Under-prediction degraded only at × 0.25 (15/20).
- **Random slip adds noise-like error.** On V4 the cliff is between 20 and 40 mm of median 700-ms error: 19/20 at 20 mm, 7/20 at 40 mm.
- **A level is kept if C1's open-loop 700-ms forecast error passes either threshold:**
  - a predicted/true ratio beyond × 1.5, the largest over-prediction tested; or
  - a random error (after removing the mean ratio) of at least 20 mm.

  One level beyond is also kept. If no stable level passes either, the harness absorbs that perturbation. That is a result in itself, and the perturbation is not run on the navigation mazes.

**Step 0 (re-measuring the curve on the new harness) is dropped** (Andrew, 3 October). The thresholds above come from the old harness, whose cliff was mostly reserve-trap stalls, so they guide the choice of levels only roughly. The characterisation's open-loop C1 error is the primary measure.

## Evaluation

- **Mazes:** the 20 preliminary mazes 30–49, the same as the sensitivity experiment, so results line up with its dose-response.
- **Setting:** recovery off (the default) with the coverage-rule fix.
- **Controllers (Andrew, 3 October):** C1, C3 and C4.
  - **C1:** unchanged command-history kinematics, fitted on nominal dynamics.
  - **C3:** the large past-frames decoder, which sees frames 0.5 s and 1 s ago.
  - **C4:** the matched supervised predictor, which sees three frames plus command history.
  - **C0 and C2 are not run.** Locomotion limits are read from the open-loop characterisation (falls, realised speed) and from failures shared by all three controllers.
- **Primary measures, per controller and condition:**
  - **Mechanism:** the forecast error the planner acted on while driving (median 700-ms error and ratio, by movement type), from the closed-loop scorer.
  - **Outcome:** success (Wilson 95%), SPL, median time, hold rate, contacts and clearance.
  - **Paired:** against C1 on the same mazes (bootstrap difference, discordant counts), and C3 against C4.
- **Reading:**
  - H1 holds if, under perturbation, C1's measured error rises well above C3's and C4's.
  - H2 holds if the driving gap follows, roughly where the sensitivity curve predicts.
  - If C0 also degrades, that share of the loss belongs to locomotion and the harness, not prediction.

## Retraining (superseded 3 October; see "Stages" below)

The draft's optional adaptation stage on uniform perturbations is replaced. Stage 1 runs zero-shot, and stage 2 (marked patches) trains on matched data.

## Stages (Andrew, 3 October)

### Friction in Genesis: the pair rule is the maximum

The solver combines the two surfaces' coefficients by taking the **maximum**. This was verified in `lewm/support_friction_challenge_development.py`, which reads the solver's per-geometry coefficients (`pair_combination='maximum'`).
- So a floor change alone does nothing while the robot's geometries stay at 1.0.
- The 15 September lower-friction trial therefore set μ on the floor **and** all 27 robot geometries.
- Walls keep their own coefficient, 1.0 in the capability scenes (read back from the solver on 3 October), so wall pairs stay nominal.

### Stage 1: uniform friction, zero-shot

- **Hook:** μ on the floor and all 27 robot geometries, installed before any physics step and read back from the solver, as in the room-return trial.
- **Levels:** picked by the open-loop characterisation above.
- **Controllers:** C1, C3 (large past-frames decoder) and C4 (its matched refit) exactly as in the new-harness re-run, with no retraining. Recovery off, harness `reserve_exit_v1`, mazes 30–49.
- **Reported:**
  - **Forecast error by movement type:** hold, rest start, in-place turn, steady cruise, steady arc and command switch. Median 700/800-ms error and median predicted/true ratio, per controller and level, from `scripts/score_go2_dev_closed_loop_prediction_development.py`.
  - **Driving outcomes:** success (Wilson 95%), SPL, time, hold rate, contacts and clearance. Paired against C1, and C3 against C4.

### Stage 2: marked low-friction patches, after a matched refit

**Why training is needed.** Neither model has ever seen the marker, so zero-shot it is just a colour on the floor. The C3 decoder and C4 are refit on matched data, with the same procedure as the decoder fix.

**Patch hook: a friction field, not floor tiles** (revised 3 October after a feasibility probe).
- **Why not tiles.** The capability scene builder turns every static object into an invisible collision box plus a patterned wall mesh, and the mapping and evaluation treat static objects as walls. Tiles would also add step edges.
- **How the field works.** Genesis multiplies each geometry's base coefficient by a runtime per-geometry ratio (`set_geoms_friction_ratio`, simulation state), then takes the maximum over the pair.
  - The floor's ratio is set to μ_p.
  - Each leg's four calf geometries (the foot sphere is one of them) get ratio 1.0 off a patch and μ_p over one. This is updated at every 20-ms policy step from the foot sphere's position.
  - So contact friction is nominal off-patch and μ_p on-patch. Body and other geometries stay at 1.0 against the floor, and walls stay nominal.
- **Probe (12-m room, 0.2 m/s commanded, patch beyond a line, μ_p = 0.2):**
  - speed is 0.18–0.21 m/s before the patch;
  - with all four feet on it, speed is 0.10–0.12 m/s, matching uniform μ = 0.2's speed ratio of 0.62;
  - a short transition follows while the legs cross the boundary.
- **The marker recolours the floor itself** (revised the same afternoon). The renderer draws exactly two static surfaces, floor and wall union, in a fixed order, so an extra mesh breaks its contract.
  - In the marked condition, the floor mesh's 12.5-cm quads whose centres lie in a patch cell are recoloured a uniform `slick_patch` blue-grey (RGB 0.30/0.40/0.55), in place of the floor's random greys.
  - Geometry, identity witnesses, depth and draw order are unchanged.
- **Placement:**
  - patches are runs of 2–3 consecutive interior cells on the shortest home-to-beacon cell route, so each covers the full corridor width;
  - start and beacon cells are never patched, and the return trip crosses the same cells;
  - the runs are seeded and cover 20–40% of the route's cells. All 20 preliminary mazes get one run, at 22–38% coverage.
- **Built and tested** (`lewm/dev_dynamics_patches_development.py`, `scripts/test_go2_dev_dynamics_patches_development.py`) on dev maze 0, with a test patch in the cell ahead:
  - 110 marked floor quads, and the marker changes 18% of the start frame ([frame](go2_navigation_dynamics_patch_marker_start_frame_2026-10-03.png));
  - every foot-floor contact's solver friction is 0.2 over the patch (506 contacts) and 1.0 off it (483);
  - the unmarked control has identical friction and no recolouring.
- **The field's ratios are written before every policy step,** because settling resets them.
- **The speed effect is measured in the open arena:** a single 1.3-m cell gives only about 2 s on the patch.

**Visibility requirement.**
- The forward RGB camera sees the floor from roughly 0.9 m ahead of the base centre. This is estimated from its mounting and its 63° vertical field of view, and is checked by render.
- So a patch must run at least 1.5 m along the route. Otherwise the marker is out of view while the feet are on it, and its entry is seen only about 1 s ahead.
- Patches are contiguous runs on route cells covering 20–40% of the route length. They are placed by a seeded rule that uses the maze geometry only.

**Recordings** (training-only layouts). New layouts come from the capability generator, excluding every prior graph, as in the on-policy round. Episode 0 only, and C1 drives each once, recovery off, with no retries.

| Role | Layouts | Use |
|---|---:|---|
| `patch_fit` | 24 | training contexts |
| `patch_heldout` | 6 | offline acceptance only |

- **Why 24.** The on-policy round's 16 C1 missions gave 25,469 contexts and moved the decoder measurably. Here only about 30% of contexts involve a patch (the robot on one at *t*, or entering one within 800 ms). 24 missions should give roughly 11,000–12,000 patch contexts among about 38,000.
- **Frames** are regenerated by verified deterministic replay. **Contexts** follow the on-policy rules, and a mission with any contact contributes nothing.

**Refit: matched pair, decoder-fix procedure.**
- The feature cache is rebuilt: the earlier pools (34,745 contexts; deleted after the decoder choice) plus up to 12,000 patch and 4,000 off-patch contexts from the new missions.
- C3's large past-frames decoder and C4 are fitted on identical data, with the `p3_large_past_frames` recipe: 3,520 updates and lr 3e-4.
- Proposed batch mix: 32 old, 16 maze, 8 on-policy, 8 patch.
- Three seeds; the median seed is chosen by the existing rule (mean of the per-type median 800-ms errors).
- **C1 is unchanged.** Its inputs (commands) carry no information about where patches are. A refit on the same mix takes minutes if wanted as a fairness check.

**Offline acceptance** (held-out patch layouts, before any closed-loop run):
- **On patch:** the 800-ms XY error and ratio of C3 and C4, each against its pre-refit version.
- **Off patch:** no worse than 1.05 × the pre-refit error, the no-loss rule used before.

**Evaluation.**
- Mazes 30–49 with patches placed by the same rule. C1, the refit C3 and the refit C4, recovery off.
- Forecast error is reported on and off patch by movement type, alongside driving outcomes.

**Recommended control.** The same patches, unmarked (plain floor texture). If the refit models gain as much there, the gain comes from motion history, not the marker.

### Decisions (Andrew, 3 October, afternoon): approved as planned

- **Storage:** the feature cache is stored as float16, after a subset-refit equivalence check. The deployed decoder's recipe is refit on a fixed subset from float16 and from float32 features, and the two must give the same per-type acceptance numbers within 0.5 mm. Otherwise this stops for Andrew.
- **The unmarked-patch control is included:** C1, C3 and C4 on mazes 30–49, with the same patches, no marker.
- **C1 is refit on the patch data** as a fairness check, using the same mix as the decoder and C4. It runs alongside unchanged C1.
- **The offline improvement check stays** before any closed-loop stage-2 run, as a sanity check.
- **Stage 1 μ rule:**
  - Choose μ in the open-loop characterisation so that the gait stays stable (no falls or stumbles) while the forecast error from commands alone is clearly above nominal.
  - The chosen μ is reported with its reason.
  - Definitions, fixed before any characterisation run, are below.

**Stage 1 μ selection, made concrete.**
- **Grid:** μ = 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15. Each level drives the fixed calibration tapes (rest starts, cruise, arcs, in-place turns, command switches) on a flat floor, 3 repeats.
- **Fall:** the base falls below 0.15 m, or a non-foot geometry touches the floor for more than 0.5 s.
- **Stumble:** any non-foot floor contact; or a base-height dip more than 5 cm below the nominal gait's minimum; or roll or pitch beyond the nominal gait's maximum plus 10°.
- **Stable:** no fall and no stumble in any tape or repeat at that μ.
- **Clearly above nominal:** C1's open-loop 700-ms forecast error from commands alone, median over the tapes, is at least twice nominal and above the nominal 95th percentile.
- **Choice:** the lowest μ that is stable with one grid step of margin (the next lower level is also stable) and clearly above nominal.
  - It is reported with its error by movement type, its realised/commanded speed and yaw rate, and the stability margin.
  - If no level qualifies, stage 1 stops for Andrew.

**Stage 1 μ chosen (3 October, 13:05): μ = 0.2**, by the pre-declared rule ([characterisation](go2_navigation_dynamics_stage1_characterisation_2026-10-03.md)).
- No level from 1.0 to 0.15 fell or stumbled, so 0.2 has its one-step margin.
- C1's commands-only 700-ms error there is 29.8 mm median, against 5.9 mm nominal (5.1×) and above the nominal p95 of 16.6 mm.
- Friction mainly cuts forward progress (speed ratio 0.62) and barely changes the turn rate (1.01).

**Stage 1 runs:** C1, C3 and C4 × mazes 30–49 at μ = 0.2, recovery off, harness `reserve_exit_v1_1`, through the v4 pinned launcher (`--dynamics friction:0.2`).
- C1 and C4 run on the CPU once the C0 gate has passed and the re-run's CPU missions are done.
- C3 runs on the GPU after the re-run's C3.

### Stage 1 launched (3 October, 17:15)

**C1 and C4:** cohort `dyn1_friction02_cpu`, mazes 30–49 at μ = 0.2, recovery off, harness `reserve_exit_v1_1`, v4 pin `e44a23d3`.
- Each mission writes `native/dynamics_friction.json`.

**C3:** cohort `dyn1_friction02_c3`, launched automatically once the re-run cohort `rexit_rerun` writes its `result.json`.
- This keeps at most two C3 missions on the GPU.

**Contact stops are outcomes.** A disallowed contact stops a mission as a `PhysicalStop`, which the cohort runner records as a technical stop with no evaluation.
- Each such mission is scored afterwards with the failure reader (`scripts/read_go2_dev_mission_development.py --reader failure`): round trip failed, contacts counted.
- The 3 October smoke mission (dev maze 0) ended this way at 219 s: rear-left calf against a wall.

### Stage 1 interim (3 October, 22:05): μ = 0.2 freezes every mission in the harness, before prediction can matter

**C1:** 0 of 20 round trips; none reached the beacon.
- 17 timed out holding.
- 3 ended on a contact (mazes 30, 33, 42).

**C4:** 0 of 9 so far; none reached the beacon.
- 7 timed out holding.
- 2 ended on a contact (mazes 31, 32).

**C3** is queued behind the re-run.

**The mechanism is the same in both** (`scripts/report_go2_reserve_exit_rerun_development.py`).
- Most stalls begin 19–36 s in, during the scripted opening look-around.
- At onset the centre is already inside the 0.45-m disc (true clearance 0.40–0.44 m) and the robot is in scan mode.
- Slip while turning in place carries the body inward much more than the nominal 5–7 cm. Inside the disc the turn reserve blocks both turns, and scan mode excludes translation, so the robot holds until the budget ends.
- The robots travel only 2–6 m in total.
- The re-run's exit rule cannot act here: translations are excluded while scanning.
- A few later stalls end with a clearance-turn latch or the coverage rule overriding an exit.

**Contacts:** all five are the rear-left calf against a wall, 209–476 s in, after 2,700–5,700 hard-clearance samples of turning or holding pressed against the wall. This is the rear-leg blind spot, worse under slip.

**Reading.**
- At this level the shared harness freezes both controllers first, so stage 1 at μ = 0.2 cannot separate C1's command-only prediction from C4's visual prediction.
- The forecast errors that differ (cruise and arcs) are never exercised, because the robots barely leave the start.

**Options (Andrew decides):**
1. **Address the residual turn block** (the candidate change from the re-run): let an in-place turn proceed inside the reserve or disc when its forecast centre path does not lose clearance. Then re-run stage 1. This touches the same rule that leaves C2 at 9/20 and C3 stuck on maze 31.
2. **Use a milder level, μ = 0.3.** It also met the stage-1 rule: C1's error was 3.2× nominal, with speed ratio 0.78. Turning drift would be smaller, but the trap may still dominate.
3. **Exempt the scripted look-around** from the turn reserve (as the stage-2 pessimistic-unknown rule was exempted). This covers only the early freezes.

### Paused (Andrew, 4 October, 00:45): all experiments stopped, zero load

**What ran.** The stage-1 tallies at the pause: C1 0/20 round trips, 0 beacons, 3 contact missions; C4 0/18 round trips, 0 beacons, 2 contact missions. The two C4 contact stops (mazes 31, 32) were scored with the failure reader.

**What was stopped.**
- Stage-1 C4 mazes 48 and 49, and stage-1 C3 mazes 30 and 31, were interrupted mid-mission.
- Their run directories are kept as incomplete records and carry no result.
- Stage-1 C3 mazes 32–49 never started.

**On resume:**
- Rerun the interrupted and unstarted missions as new cohorts through the v4 pinned launcher. Cohort directories are never reused.
- Before that, Andrew's open decisions may change what runs:
  - the turn-reserve fix, which would make the μ = 0.2 stage-1 results a pre-fix record;
  - the stage-2 speed guard.

**Resumed (4 October, 09:20).** Both batches launched through the v4 pinned launcher (pin `cfe3d0ec`; runtime files unchanged since `1cac1bc5`), still on harness `reserve_exit_v1_1`:
- `dyn1_friction02_c4b`: C4 mazes 48–49.
- `dyn1_friction02_c3b`: C3 mazes 30–49, two GPU lanes.

The interrupted directories in `dyn1_friction02_cpu` (C4 48–49) and `dyn1_friction02_c3` (C3 30–31) stay as incomplete records. Andrew's two decisions remain open.

**Stage-1 C4 complete (4 October, 10:25): 0 of 20 round trips and no beacon reached, the same as C1 (0/20).**
- Two C4 missions ended on contact (mazes 31, 32).
- Maze 48 froze holding.
- Maze 49 froze without contact but held 0.7 cm from a wall: 56,751 samples inside the 2-cm operating bound; the 5-mm hard bound was not crossed.
- C3 (`dyn1_friction02_c3b`) is running.

### Decisions (Andrew, 4 October, evening)

1. **The in-place turn fix is approved.** A turn in place is allowed inside the margin when its predicted centre path does not lose clearance. It goes into a new harness version, with the gates as before: C1 on dev mazes 0–9 × episodes 0–1, then C0 at least 19/20.
2. **The remaining μ = 0.2 C3 missions are cancelled.** The 10 completed are kept for the forecast-error table.
3. **The stage-1 level rule is re-picked.** "Lowest stable μ" was the wrong objective. The rule is now: the highest (mildest) μ on the calibration grid at which C1's commands-only error is clearly above nominal (at least 2× and above the nominal p95), and also the next lower level.
   - From the [characterisation](go2_navigation_dynamics_stage1_characterisation_2026-10-03.md), that gives **μ = 0.3** (18.6 mm, against 2 × 5.9 mm and the 16.6-mm p95) **and μ = 0.25** (22.1 mm). Both are stable.
   - μ = 0.4 fails the rule: 13.4 mm is below the nominal p95.
   - Stage 1 runs at both levels on the fixed harness, with C1, C3 and C4, recovery off. It reports forecast error by movement type and outcomes.
4. **Stage 2 uses μ_p = 0.3,** the chosen stage-1 level. Only if patch edges still trip the 0.3-m/s guard is it raised to 0.40 m/s, for all controllers on patch missions.
   - The 3 October smoke missions at μ_p = 0.3, on the version without the turn fix, tripped it in 2 of 3. This is re-checked with a stage-2 smoke on the fixed harness before any recording.
5. **The shared page is published now with the clean baseline only.** Dynamics stays out until the re-run.

The μ = 0.2 rear-left-calf contacts are noted in the known limitations, with no follow-up experiments.

### Run order (Andrew, 4 October, 19:10)

1. **No C3 normal-friction reference on `reserve_exit_v2` for now.** A 10-maze version runs later only if the stage-1 result needs it. C1, C2 and C4 at normal friction on v2 run on the CPU as the same-harness reference.
2. **After C3 at μ = 0.3 finishes,** send a short interim: forecast error by movement type plus outcomes, C1, C3 and C4. Then wait for Andrew's go before C3 at μ = 0.25. C1 and C4 at μ = 0.25 run anyway on the CPU.
3. **If Andrew is slow to answer,** use the GPU for stage 2's patch smoke test and the feature-cache build rather than leaving it idle.

### Decisions (Andrew, 5 October, morning)

1. **C3 at μ = 0.25 is skipped.** C1 and C4 there collapse to 1/20 on turn-exit contacts.
2. **The turn-exit contacts are a recorded limitation,** with no harness change mid-experiment (see the known limitations). v2 also produced a normal-friction contact (C2), so a v3 turn check on the legs' swept footprint is required before the confirmatory sealed run. It is not built now.
3. **Stage 2: go.** The smoke test and the feature-cache build start once C3 at μ = 0.3 finishes. Three design changes:
   - **Placement:** patches go on straight route segments, at least 0.5 m from junctions and dead ends, to minimise in-place turns on patches. This changes the 3 October placement rule; the visibility requirement (at least 1.5 m along the route) stands.
   - **Primary measure:** closed-loop forecast error on decisions, binned by distance to the patch edge (approach, entry, on-patch), marked versus unmarked, per controller. Mission outcomes are secondary.
   - **An adaptive C1 baseline (C1A):** C1's forecast scaled by the recent tracked-versus-predicted travel ratio. It uses the tracker pose only, with no privileged data and no training. This separates reacting to slip from anticipating it.
4. **The full stage-1 interim** is sent when C3 at μ = 0.3 finishes.

### Decisions (Andrew, 5 October, late morning): stage-2 layouts and C1A

**Placement constraint found.** Under the 5 October placement rule (straight segments, at least 0.5 m from junctions, dead ends and endpoints, and at least 1.5 m long), only mazes 33, 42 and 47 of 30–49 can hold a patch. Across all 100 capability-maze episodes (dev, validation and prelim), 14 can hold one and 8 can reach 20% coverage.

**Stage-2 layout sets: a selected family, "routes with long straights".**
- New sets come from the capability generator, through the registration procedure, disjoint from every existing graph: 24 fit, 6 held-out and 20 evaluation.
- Only episodes whose route has a usable straight segment are kept, and placement must reach the 20% coverage minimum.
- They are a **selected family**, not a sample of the capability distribution. Results on them describe routes with long straights.
- The smoke test runs on prelim mazes 33, 42 and 47 as planned.

**No-patch reference** on the 20 evaluation mazes, with C1, C4 and C3. It is secondary, for outcome context.

**C1A parameters, fixed before any stage-2 run** (`lewm/dev_c1_adaptive_travel_v2_development.py`, entry v7):
- **Two ratios:** translation (tracked over predicted XY travel) and rotation (tracked over predicted heading change). Both compare the tracker pose with C1's own forecast, re-run on the commands the controller requested over the past 0.4–0.8 s.
- **Samples come only from meaningful commanded motion:** all-hold windows are excluded, and a sample needs at least 2 cm (translation) or 2° (rotation) of predicted motion.
- **Each ratio:** the median over a 3-s window, 1.0 until there are two samples, clipped to 0.3–1.5.
- **Applied to every candidate and horizon:** XY times the translation ratio, yaw times the rotation ratio.
- **Inputs:** tracker pose and the controller's own requested commands only; no privileged data, no training.
- **Version history:** the first version (`lewm/dev_c1_adaptive_travel_development.py`, one XY ratio) is superseded before use. Its normal-friction check, `c1a_check` on dev mazes 0–9, is kept as a functional check only.
- **Validation:** C1A v2 at uniform μ = 0.3 on mazes 30–49 (CPU), with forecast error by movement type against C1's 19 mm.

### C1A v2 validation at uniform μ = 0.3 (5 October, 10:30)

Cohort `c1a2_mu030`: C1A v2 on prelim mazes 30–49, recovery off, harness `reserve_exit_v2`, v7 pin. PRELIMINARY.

- **Outcomes:** 15/20 round trips (C1: 16/20). 5 contact stops (mazes 31, 33, 36, 40, 44), all scored by the failure reader, against C1's 4. Completed runs had 0 contacts and a minimum clearance of 5.7 cm.
- **Forecast error** (median 700-ms XY error, predicted/true ratio):

| | All | Cruise | Turn | Arc | Switch |
|---|---|---|---|---|---|
| C1, μ = 0.3 | 19 mm ×0.92 | 40 mm ×1.17 | 12 mm ×0.54 | 21 mm ×0.91 | 25 mm ×1.01 |
| **C1A v2, μ = 0.3** | **18 mm ×0.89** | **37 mm ×1.11** | 12 mm ×0.55 | 20 mm ×0.92 | 25 mm ×0.98 |
| C1, normal friction (v2 reference) | 6 mm ×0.96 | 4 mm ×1.00 | 5 mm ×0.75 | 7 mm ×0.95 | 7 mm ×0.96 |

- **Why the gain is small.**
  - Uniform low friction does not scale C1's error uniformly: cruise is over-predicted (×1.17), turning translation is under-predicted (×0.55), arcs are ×0.91 and switches about ×1.0.
  - The ratio samples mix these movements. Straight-travel samples (predicted at least 8 cm) have a median of 1.00; turn, arc and start samples a median of 0.70.
  - So the applied translation ratio stays near 1.0: median 1.00, 10th to 90th percentile 0.73–1.14, and 1.0 for 26% of decisions (too few recent samples).
  - The rotation ratio is a median 1.05: low friction barely changes the turn rate.
- **The tracker is not the cause.** Over 800-ms windows, tracker travel equals physics truth (median ratio 1.00, quartiles 1.00) at both μ = 0.3 and normal friction.
- **Parameters unchanged,** as fixed. At patch edges the slip is transient and localised, a different regime from uniform friction.

### Stage-2 layout sets registered (5 October, 10:40): a selected family, "routes with long straights"

**Script:** `scripts/register_go2_stage2_patch_sets_development.py`. Registry: `<capability root>/stage2_sets_v1_registry.json`, sha256 `f7019303…`.
- **Generator and checks:** the capability generator's candidate and acceptance rules, `make_spec` and episode rules, with structural checks only.
- **Exclusions:** the capability prior graphs, all 90 capability layouts, `c3v2_sets_v1` and `c3v3_sets_v1`.
- **Selection:** every unique accepted layout within the 10,000-candidate bound is built in order, with episodes 0 and 1. A layout qualifies when its first such episode reaches at least 20% patch coverage under the v2 placement rule.
- **Seed attempts:** construction seeds were tried in fixed order, 2026100501, 2026100502, 2026100503 and 2026100504. They gave 47, 45, 43 and 50 qualifying layouts from about 890 unique ones each; attempt 4 is used. The choice depends on structural counts only.
- **Sets:**

| Set | Mazes | Episodes |
|---|---|---|
| `stage2_fit` | 0–23 | one each |
| `stage2_heldout` | 24–29 | one each |
| `stage2_eval` | 30–49 | one each |

  26 sets use episode 0 and 24 use episode 1. Coverage is 20.5–39.7%, with 1–2 strips per route.
- **Structural checks:** all passed (unique topologies and embeddings, disjoint from the excluded graphs, episode rules, no physics or rendering). Hash readback matches.
- **Loading:** missions load these sets through `scripts/run_go2_dev_mission_stage2_development.py`, with v8 entry and launcher. All 50 load hash-verified, and runtime placement reproduces the registered coverage.
- **Not checked against `sealed_test_v2`.** Its seeds are private and its folder is sealed. The generator's topology space is small: about 890 unique layouts per 10,000 candidates, and 40% of sealed_test_v2's own candidates hit a prior topology. So a few of these 50 may share a topology with a sealed_test_v2 maze.
  - **This must be settled before any of these layouts trains a model** (the stage-2 refit).
  - Options: a custodian-side overlap check, or regenerating the rigorous-phase set just before that phase, excluding every graph built by then.
  - Evaluation runs on these sets do not train anything.

### Stage-2 smoke test (5 October, 11:30) and patches v3

Cohort `s2smoke_mup030`: C1, marked v2 strips, μ_p = 0.3, prelim mazes 33, 42 and 47, harness `reserve_exit_v2`, v8 pin. PRELIMINARY.

- **The guard still trips at strip edges.** Mazes 42 and 47 stopped on `CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP` at 0.300–0.301 m/s, with the body centre 0.13–0.19 m inside the strip, just after entry. (The 0.82-m/s peak at 0.1 s in every run is the spawn drop, outside the guard.)
  - By Andrew's 4 October rule, the guard's speed limit is raised to 0.40 m/s on patch missions, for all controllers, marked and unmarked.
  - Non-foot ground contact and domain stops are unchanged.
- **The uniform marker starved the visual tracker (maze 33).**
  - The robot's centre spent 228 of 306 s on the strip, mostly scanning: 499 left turns, with `ADDITIONAL_VIEW_REQUIRED` 500 times and `LOW_VISUAL_SUPPORT` 14 times. It ended on "measured visual pose unavailable", with 0 contacts.
  - On the strip, the weaker camera's selected features had a median of 66 and a 10th percentile of 0, against 119 off it. The same corridor at uniform μ = 0.3 without a marker never starved (median 120, 10th percentile 86).
  - A uniform colour erases the floor's texture. It would also confound marked with unmarked, since unmarked strips keep their texture.
- **Patches v3** (`lewm/dev_dynamics_patches_v3_development.py`, v9 entry and launcher, `--dynamics patches3:MU:marked|unmarked`) changes two things:
  - a **tinted** marker: each floor quad keeps its own colour scaled by (0.45, 0.70, 1.00), so texture survives and the strip reads blue;
  - the **0.40-m/s guard**.
  - Placement and the friction field are v2's. 3 synthetic tests pass.
- **Re-run:** `s2smoke3_mup030`, marked on 33, 42 and 47, plus unmarked on 33 as a texture control.

### Stage 1 closed (Andrew, 5 October, midday)

**Result.**
- No zero-shot visual advantage.
- Outcome differences between controllers are within noise.
- The contact stops come from the v2 turn exit.

Details are in [the stage-1 results](go2_navigation_dynamics_stage1_v2_results_2026-10-05.md). PRELIMINARY: prelim mazes 30–49, recovery off, harness `reserve_exit_v2`.

### Decisions (Andrew, 5 October, midday)

1. **Contamination, option 2.** The rigorous-phase sealed set is regenerated immediately before that phase, excluding every graph built by then. The current `sealed_test_v2` stays untouched and unseen, and will be replaced then. Recordings, the cache and the refits proceed on the stage-2 layouts.
2. **The tinted marker is approved.** Before the refits, a **marker-visibility probe** runs: a linear classifier on frozen V-JEPA features of tinted versus untinted floor frames, tested on held-out mazes. Its accuracy is reported. **If it is near chance, stop and report before the refits.**
3. **C1A, one revision (v3), designed on stage-1 data only.** Ratios are kept per movement type (cruise, arc, turn, switch), each from that type's recent decisions, with the same window and clip rules. It is re-validated at uniform μ = 0.3, then frozen and recorded here before any stage-2 evaluation.
4. Stage 1 is closed (above).

### Correction: the strip, not the marker, starves the tracker (smoke re-run, 5 October, 12:30)

Re-run `s2smoke3_marked` (C1, patches v3, μ_p = 0.3, marked, mazes 33, 42 and 47) and `s2smoke3_unmarked` (maze 33). PRELIMINARY.

- **The guard is fixed.** Peaks within 0.5 m of a strip edge were 0.27–0.29 m/s, below even the old 0.3-m/s limit this time; there were no guard stops. Every run had 0 contacts.
- **Maze 42 (marked):** round trip, 32 s on the strip.
- **Mazes 33 and 47 (marked) and maze 33 (unmarked):** the centre stayed on the strip for 228–266 s, and each mission ended on "measured visual pose unavailable".
  - On the strip, the weaker camera's selected features were a median of 66–73 in maze 33 (both conditions), with a 10th percentile of 0. Maze 47 had a median of 132 and a 10th percentile of 28. Off the strip they were a median of 120–122.
- **My earlier diagnosis was wrong.** The unmarked control starves the tracker exactly as the uniform marker did (median 66, 10th percentile 0), so the textureless marker was not the cause. The tinted marker stays (approved, and harmless).
- **What the stall looks like:**
  - Requested commands on the strip were mostly hold: 81% in maze 33 and 63% in maze 47. There was little forward motion (7% and 26%) and some turning (12%).
  - Maze 47 covered an 18-m path for a 0.30-m net move, turning 45 rad in total.
  - The body stays upright (roll at most 6.4°, pitch 3.3°, base height at least 0.30 m).
  - The same maze 33 was a clean round trip at uniform μ = 0.3, so the stall is specific to a local low-friction strip.
- **Mechanism not yet known.** The note in `lewm/dev_dynamics_patches_v3_development.py` attributing the starvation to the marker is wrong. That file is pinned by running cohorts, so it is corrected here and will be corrected in the file once no v9 or v10 batch can launch.

### Stage-2 recordings (Andrew, 5 October, 12:15): option 2 with a stall stop

- **Decision.** Record now, accepting stalls.
  - A recording mission ends once it has been stalled (no translation) for 60 s.
  - The training set is built only from decisions with meaningful motion: holds and latched-veto ticks are excluded.
  - Patch-approach and patch-entry contexts per mission are reported.
  - Evaluation design unchanged: forecast error by distance to the patch edge is primary; outcomes are secondary, with stalls classified.
  - No milder strips.
- **Strip-stall diagnosis:** a real wall, not a phantom ([diagnosis](go2_navigation_stage2_strip_stall_diagnosis_2026-10-05.md)).
- **Stall stop v1 was wrong, and the first cohort was stopped.**
  - `s2rec` (v11 pin) used a stall stop that compared only the positions 60 s apart.
  - 4 of its first 5 recordings were cut while the robot was driving loops (5–15 m of path, 47–72% forward or arc commands) that returned to an earlier spot. The fifth ended on a contact.
  - The cohort was stopped (launcher and missions). Its 5 finished runs and any partial ones are kept and labelled superseded.
  - **Stall stop v2** (`lewm/dev_recording_stall_stop_v2_development.py`, v12 pin) counts a stall only when the base stays within 0.10 m of its window-start position for the whole 60 s.
  - All 30 recordings are relaunched as `s2rec2`: C1, marked strips (patches3, μ_p = 0.3), stage-2 fit and held-out mazes with their registered episodes, recovery off.

### C1A v3 validated and frozen (5 October, 12:45)

Cohort `c1a3_mu030`: C1A v3 on prelim mazes 30–49, uniform μ = 0.3, recovery off, harness `reserve_exit_v2`, v10 pin. PRELIMINARY.

- **Outcomes:** 15/20 round trips. 4 contact stops (mazes 31, 35, 36, 44), all scored by the failure reader, and 1 other failure (maze 33). Completed runs had 0 contacts and a minimum clearance of 3.4 cm. For comparison: C1 16/20, C1A v2 15/20.
- **Forecast error** (median 700-ms XY error, predicted/true):

| | All | Cruise | Turn | Arc | Switch |
|---|---|---|---|---|---|
| C1 | 19 mm ×0.92 | 40 mm ×1.17 | 12 mm ×0.54 | 21 mm ×0.91 | 25 mm ×1.01 |
| C1A v2 | 18 mm ×0.89 | 37 mm ×1.11 | 12 mm ×0.55 | 20 mm ×0.92 | 25 mm ×0.98 |
| **C1A v3** | **18 mm ×0.92** | **39 mm ×1.14** | 12 mm ×0.54 | 21 mm ×0.95 | 24 mm ×1.00 |

- **Why per-type ratios barely engage.**
  - With a 3-s window and two samples needed, each type rarely has enough recent samples: the applied translation ratio is 1.0 for 91% of decisions (cruise), 81% (arc) and 36% (switch).
  - Turns never adapt translation, because C1 predicts almost no translation for an in-place turn and the 2-cm rule admits no sample. The turn error (×0.54) is slip drift added on top of a near-zero prediction, which a multiplicative ratio cannot correct.
  - Rotation ratios stay at 1.0–1.02: low friction barely changes the turn rate.
- **Frozen.** C1A for all stage-2 evaluation is **v3** (`lewm/dev_c1_adaptive_travel_v3_development.py`), launched through the v10 entry or later. Its parameters, fixed before any stage-2 evaluation:
  - types: cruise, arc_steady, turn and switch; hold and rest_start are not adapted;
  - one translation ratio and one rotation ratio per type, each tracked over predicted;
  - samples: no all-hold windows, at least 2 cm or 2° predicted, a re-forecast window of 0.4–0.8 s;
  - each ratio: the median over 3 s, 1.0 until two samples, clipped to 0.3–1.5;
  - inputs: tracker pose and the controller's own requested commands only.
  - At patch edges, slip is localised and transient; whether C1A reacts there is part of the stage-2 measurement.

### Stage-2 recordings done (5 October, 13:40): `s2rec2`

C1, marked strips (patches3, μ_p = 0.3), stage-2 fit (0–23) and held-out (24–29) mazes, recovery off, harness `reserve_exit_v2`, stall stop v2, v12 pin. PRELIMINARY. Counts come from `scripts/count_go2_stage2_recording_contexts_development.py`.

**Endings (30 missions):**

| Ending | Count | Notes |
|---|---|---|
| Round trip | 15 | |
| Contact stop | 6 | all scored by the failure reader; all are the rear-left calf during a turn-exit in-place turn, 5 on a strip and 1 0.17 m outside |
| Stall stop, strip trap | 5 | centre within 0.45 m of a wall, dispatch veto latched |
| Stall stop, other | 3 | |
| Other failure | 1 | |

**Usable contexts.**
- A context is usable when its 700-ms window has meaningful requested motion and no vetoed or latched dispatch tick.
- Under the on-policy rule, the 6 contact missions contribute nothing.
- From the 24 contributing missions: **8,521 usable contexts.**

| Bin | Contexts |
|---|---|
| approach (0.3–1.5 m, moving in) | 1,998 |
| entry (±0.3 m, moving in) | 583 |
| on patch (at least 0.3 m inside) | 892 |
| exit | 619 |
| off patch | 4,429 |

- **Per mission:** approach 31–171, entry 14–67 (highest in strip traps, which hover at the edge), on patch 0–63.
- **Missions reaching no strip:** held-out 25 and 26 (stalled before any strip) and fit 14 (failed early).
- **Held-out mazes with strip contexts:** 27, 28 and 29: 89 entry and 116 on-patch contexts in all.

**Against the plan's estimate.** The plan expected about 11,000–12,000 patch contexts. This gives about 4,100 patch-related ones (approach, entry, on patch and exit), plus 4,400 off-patch, because missions are shorter (stalls and stops) and contexts are filtered for motion. The refit's patch share of the batch mix will be correspondingly smaller unless more recordings are added.

### Recording replays and marker-visibility probe (5 October, 17:45)

**Replays.** All 24 contributing missions were re-simulated by `scripts/replay_go2_stage2_recording_frames_development.py` and verified bit-exact:
- consumed-packet hashes, selected actions, C1 forecasts, dispatch commands and reasons, applied commands, native trace values and published poses;
- maximum position and yaw error 0.

Fit 14 ended in a tracking loss ("measured visual pose unavailable" at 119.5 s). That failure came on a frame acquired one tick after the last logged request, so the replay needed a fourth rule: acquire and check that final frame after the loop, and compare published poses as a prefix (commit b75bb37c). The first, failed attempt is kept as `failed_attempt1_*`.

**Probe** (Andrew, midday decision 2). `scripts/probe_go2_marker_visibility_development.py`; result in `<capability root>/stage2_marker_visibility_probe_v1/result.json`. Method as fixed in the script before it ran.

- **Frames:** every second replay frame, 18,812 in all, labelled from pixels: tinted (at least 2% tinted pixels) or untinted (none).
  - Train, fit mazes: 2,553 tinted and 12,430 untinted.
  - Test, held-out mazes 25–29: 396 tinted (only on mazes 27–29) and 3,433 untinted.
- **Frozen V-JEPA** (mean and max pooled, 2,048 values; logistic regression, λ = 1e-3), on the held-out mazes:
  - accuracy 0.998;
  - **balanced accuracy 0.994** (chance 0.5);
  - AUC 0.99998;
  - per maze 0.996–1.000; train accuracy 1.0.
- **Pixel reference** (mean colour ratios, 6 values): balanced accuracy 0.884, AUC 0.9995.
- **Verdict:** not near chance. The marker is linearly readable from the frozen features, so the cache and refits proceed as decided.
- **Caveats** (not tested further):
  1. Frames with 0–2% tinted pixels are excluded, so this tests clear views of a strip, not distant or marginal ones.
  2. Tinted frames come only from strips on straight segments, so corridor context is partly confounded with the tint. The probe shows the information is available; it does not show that the tint alone carries it.
- Wall time 2.9 h, on a GPU shared with C3 missions.

### Stage-2 cache and refit design, fixed before fitting (5 October, 18:30)

**Cache** (`scripts/build_go2_stage2_feature_cache_development.py`, `<capability root>/stage2_feature_cache_v1`, launched 18:07, about 6 h of GPU):
- The decoder fix's 34,745 contexts are reproduced by its own `collect()` and checked item-for-item against the surviving `dev_c3_cache_v1/items.json`.
- To these are added the 8,520 usable `s2rec2` decisions, under the counting script's motion and veto rule (one more fails the causal-context check):

| Set | Group | Contexts |
|---|---|---|
| train (fit mazes) | `patch` (approach 1,645, entry 494, on patch 776, exit 538) | 3,453 |
| train (fit mazes) | `patch_off` (all 3,438; the 4,000 cap does not bind) | 3,438 |
| eval_patch (held-out mazes 24–29; 500 and 800 ms) | approach 352, entry 89, on patch 116, exit 81, off patch 991 | 1,629 |

- Encoding is the decoder fix's code, unchanged.
- The arrays take 67 GB in float16. Free space was 456 GB.

**C3 decoder and C4** (`scripts/fit_go2_stage2_decoder_development.py`). The `p3_large_past_frames` recipe, unchanged: past_frames variant, proj 64, hidden 896, depth 1; 3,520 updates each; lr 3e-4; seeds 2026092205, 2026093011 and 2026093012; the median seed chosen by the existing rule (eval_onpolicy, 800 ms).
- **Batch mix (64):** old 32, maze 16, on-policy 8 (evenly over the six movement types, as before), patch 8.
  - The patch 8 is split 6 : 2 between `patch` and `patch_off`, the plan's 12,000 : 4,000 ratio, uniform within each.
- **Pre-refit baseline:** the deployed checkpoint (`p3_large_past_frames_s2026093011.pt`: the C3 readout and its matched C4) scored on the same cache.
- **Offline acceptance**, as planned: on patch, the 800-ms XY error and ratio of each refit model against its own baseline; off patch (eval_patch off_patch, and the decoder fix's four sets), no worse than 1.05 × the baseline error.

**C1 fairness refit** (`scripts/fit_go2_stage2_c1_refit_development.py`, `<capability root>/stage2_c1_refit_v1`). Done, on CPU.
- **Form:** exactly C1's form, a per-horizon ridge with penalty 1 on 447 command features (prospective commands, nominal integration, and the 420-value applied-command history of the last four frames).
- **Data:** the same cache contexts, with the executed tape as the prospective commands.
- **Weights:** each context's expected draw count under the decoder mix over that fit's 3,520 × 64 draws.
- **Check:** the features rebuilt from each recording reproduce C1's logged forecasts exactly (1,895 decisions, maximum difference 0.0).
- **Median 800-ms XY error (mm), deployed C1 → refit:**

| Set | All | On patch | Entry | Exit | Approach | Off patch | Cruise | Switch |
|---|---|---|---|---|---|---|---|---|
| eval_patch | 6 → 7 | 29 → 24 | 22 → 21 | 32 → 27 | 6 → 7 | 6 → 6 | 7 → 9 | 9 → 11 |
| eval_onpolicy (no patches) | 6 → 7 | – | – | – | – | – | 5 → 8 | 8 → 9 |

- **Reading:** the refit learns an average slowing. With no information about where patches are, it gains on patches and loses on clean floor. This is the fairness reference the refit C3 and C4 must beat.

### Storage correction and retirement (4 October, evening)

**Correction.** The decoder fix's feature cache was already stored in float16 (`frames.f16`, `pred.f16`), so its 46.7 GiB was the float16 size. The earlier estimate, about 69 GiB in float32 and about 35 GiB in float16, was wrong: the stage-2 cache needs **about 69 GiB in float16**, and the float16 equivalence check is moot.

**Space freed.** Andrew approved option 1: depth-only retirement of four more closed-programme roots (see the retention policy). RecoveryStorage went from 45.75 to 85.15 GiB free.
- After about 25–30 GB of queued missions, about 55 GiB remains. That is enough for the cache only if the evaluation sets are trimmed or the queued runs finish first. The cache build is sized against free space at launch.

### Open decision: the session speed guard trips at patch edges (3 October, 15:00)

**The guard.** The frozen session stops a mission ("evaluator-only native guard", `CONTEXT_NATIVE_CONTACT_SPEED_OR_DOMAIN_STOP`) on:
- any non-foot ground contact;
- leaving the floor domain;
- **3-D base speed above 0.3 m/s.**

**Normal walking at 0.2 m/s already peaks at 0.27 m/s.** At a patch edge, the feet still on 1.0 friction push while the others slip, and the body surges.

**Measurements.**

| Condition | Peak 3-D base speed (m/s) | Guard result |
|---|---|---|
| Open arena, no patch | 0.27 | – |
| Open arena, uniform μ = 0.2 | 0.17–0.20 | – |
| Open arena, sharp edge into μ_p = 0.2 | 0.29–0.31 | – |
| Open arena, sharp edge into μ_p = 0.3 | 0.275–0.285 | – |
| Open arena, 0.3-m ramp into μ_p = 0.2 | 0.30–0.31 | – |
| C1 missions (v4 smoke), μ_p = 0.2, dev maze 0 | – | stopped 0.4 s after entering the patch |
| C1 missions (v4 smoke), μ_p = 0.3, dev mazes 0–2 | 0.302 at the trips; maze 2 peaked at 0.299 | 2 of 3 stopped within 1 s of entering a patch; maze 2 completed, 53 s on patches |

Arena values are three headings each. Uniform friction (stage 1) never trips the guard: a 219-s μ = 0.2 mission ended on a wall contact, not the guard.

**Consequence.** No μ_p that is clearly above nominal by the stage-1 rule (μ ≤ 0.3) avoids the guard. Left as is, most stage-2 missions would end at the first patch edge, and stage 2 would measure guard trips, not navigation.

**Options (Andrew decides):**
1. **(Recommended)** For stage-2 patch missions only, raise the speed limit of this guard to 0.40 m/s for every controller, marked and unmarked.
   - Keep the non-foot-ground-contact and domain stops and all disallowed-contact stops unchanged.
   - Record every guard row's peak speed, so surges are reported.
   - Use μ_p = 0.2, the stage-1 level, for a consistent dose.
   - Missions below 0.3 m/s behave identically.
2. Keep the guard and count its stops as failures.
3. Lower the patch contrast (μ_p ≥ 0.4). C1's command-only error would then not be clearly above nominal by the stage-1 rule.

### Estimate (from the decoder fix's measured costs)

| Step | Basis | Estimate |
|---|---|---|
| Patch hook, characterisation and visibility check | engineering | about 1 day |
| 30 C1 recordings | 22 on-policy missions took 54 min on CPU; +50% for slip | about 1.5 h CPU |
| Frame replay | 22 missions in about 1 h with 3 drivers | about 1.5 h |
| Feature cache | 34,745 contexts in 5.0 h (0.52 s each) on GPU | 5.0 h (earlier pools) + 2.3 h (16,000 new) = **about 7.3 h GPU** |
| Fits | 144 s per seed (decoder and C4 together) | under 15 min for 3 seeds |
| Offline acceptance | cached features | minutes |
| Stage 2 evaluation | C3 about 9–10 h GPU per 20 missions; C1 and C4 about 3 h CPU | about 10 h GPU, 3 h CPU |
| Unmarked control (included, 3 October) | same as evaluation | about 10 h GPU, 3 h CPU |

- **Total for stage 2, with the unmarked control:** about 1 day of engineering, about 27–28 h of GPU and about 9 h of CPU.
- **Storage.**
  - The cache took 46.7 GiB for 34,745 contexts (1.38 MB each), so about 69 GiB for 50,000 contexts. It is deleted after the fit.
  - RecoveryStorage had 70.6 GiB free on 3 October, less the re-run's outputs, and the 12-GiB reserve applies. Storing the cache as float16 halves it to about 35 GiB.
  - Float16 needs an exactness check first: the deployed decoder refit on float16 against float32 features, on a subset.
  - The alternative is further retirement, which needs Andrew's approval.

## Cost (based on the preliminary run's throughput)

| Step | Engineering | Compute |
|---|---|---|
| Perturbation hooks (spec override for friction; mass and COM shift and gain/torque scaling bound into session setup), with checks that the harness validators and replay still pass | about 1 day | — |
| Open-loop characterisation of about 9 candidate levels, stability checks | — | about 2 h CPU |
| Stage 1: 3 perturbations × 1–2 levels × C0–C4 × 20 mazes | — | C3 dominates at 0.8–1.2 h per mission (20 C3 missions per condition is about 8–10 h on the one GPU). 3 conditions is about 1.5 days; 6 is about 3 days. C0, C1, C2 and C4 add about 0.5 day. |
| Stage 2 (optional): recordings, cache, refits, re-evaluation | about 0.5 day | about 2 days |
| Mixed within-episode patches (stage 2): textured slick patches in the scene builder | about 1 day | about 1 day per evaluation |

**Recommended first step** (about 2 days in total):
- Hooks, the characterisation, then **friction only** at the two levels picked by characterisation, zero-shot, on C1, C3 and C4 × 20 mazes. Low-friction patches with the visual marker follow.
- Friction is the simplest to apply (one spec field, already used in this project). It acts on both translation and turning, and it fails the command-to-motion assumption most clearly.

## Risks

- The locomotion policy may be unstable on low friction or with heavy payloads. The characterisation filters such levels out, and falls are counted as failures, not hidden.
- The harness's own safety margins (clearance reserves, a 0.48-m requirement, replanning every 400 ms) may absorb even large forecast errors. The sensitivity experiment tells us how much before we spend compute here.
- C3 and C4 were trained only on nominal dynamics. A null zero-shot result does not show that vision cannot help, only that these models did not learn to use it for that. Stage 2 addresses this.
