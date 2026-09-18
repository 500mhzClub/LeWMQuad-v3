# Fixed prediction controls on the four exposed development layouts

Reuse the completed supervised controller's frozen checkpoint, persistent
visual tracker/recovery, observed mapping/routing memory, command windows,
2-mm synthetic depth noise, ideal gyro and physical arrival requirements.
No new environment types, training, controller tuning or replacement attempts.

Run three existing selectors on each of the same four development layouts:

| Condition | Action ranking | Forecast-dependent selection checks |
| --- | --- | --- |
| instantaneous | Current distance/heading derivative | Retained, including terminal overrides |
| reserved_off | Current derivative with terminal position priority | Removed; current 0.48-m clearance for movement |
| reactive_feedback | Nearest existing action to measured waypoint feedback; terminal heading alignment then forward pulses | Removed; original 0.45-m current-clearance rule |

The last control carries forward the selector/terminal rule that completed
both older short-pulse mazes. Including it avoids relying only on the
reserve-only selector, which previously deadlocked below its 0.48-m threshold.
It remains a different controller package: its margin, recovery and terminal
rules differ from full predictive planning. Do not interpret either off
comparison as isolated learned-model necessity. The instantaneous condition
isolates the main ranking change more narrowly, but still consumes forecasts
for feasibility/recovery and terminal decisions.

All conditions compute the same supervised model and forecast alternatives.
Off selectors discard their values; model validity remains checked. Total
compute differs because predictive selection checks are skipped. Current
sensor dispatch safeguards and their requested-speed stopping projection
remain in every condition, so "off" does not mean all geometric projection
is removed. Physical evaluation remains independent of runtime observations.

Twelve fixed assignments rotate condition order by layout: layout 0 runs
instantaneous/reserved_off/reactive_feedback; layout 1 runs
reserved_off/reactive_feedback/instantaneous; layout 2 runs
reactive_feedback/instantaneous/reserved_off; layout 3 repeats layout 0's order.
One native simulation runs at a time on the same per-layout CPU groups as the
completed comparison. Record hardware/resources at launch and preserve every
failure. Evaluate only after complete persistence and owner exit. Retire
completed diagnosed depth under the standing policy; retain active failure
inputs. No controller or model changes between these assigned missions.

Compare against all sixteen preceding outcomes, especially supervised and
pose/command 4/4 round trips. These are four exposed layouts in one family,
one training seed and one execution per condition/layout. Timing, ideal sensing,
controller-development preference and limited replication remain constraints.
JEPA/layout-3 raw failure stays retained for later bounded diagnosis; do not
change the tracker during this comparison. Memory attribution and realistic
sensing/timing/hardware evidence remain outstanding.

Launcher: `scripts.run_go2_persistent_visual_baselines_development`.
Plan: `docs/go2_persistent_visual_baselines_plan_2026-09-16.json`.
Roots: `go2_persistent_visual_baselines_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001`.
Thirteen focused selector tests passed, including adversarial-forecast
independence for both off runtimes and retained reactive terminal/view rules.

Plan prepared before any baseline launch. Assignment 1, instantaneous/layout 0,
launched in session 52527, owner PID 3949944. Native launch and live owner
confirmed. Preflight: 0.4% CPU busy, 68.83 GB available RAM,
7.2 GB artifact space; GPUs idle and no competing native mission. CPUs
0–7,16–23, software EGL. Completed old instantaneous/layout-1 survey-stall
depth retired after diagnosis; all its failure evidence remains.

Assignment 1 completed after owner exit and persistence. Instantaneous/layout 0
passed physical arrivals and actual selector/model/dispatch checks: round trip
198.86 simulated seconds, zero contacts, 355 on-time and 133 late plans.
170 of 488 plans had different forecast and instantaneous main preferences;
4 final actions differed from the instantaneous preference after retained
predictive selection checks. Path 15.132 m, final home distance 21.782 mm.
This is one successful main-ranking ablation, not a prediction-free controller.
Next: reserved_off/layout 0, then reactive_feedback/layout 0.

Assignment 2, reserved_off/layout 0, launched in session 77289, owner PID
3950739. Native launch and live owner confirmed. Preflight: 0.2% CPU busy,
68.9 GB available RAM, 5.71 GB artifact space; GPUs idle, no competing
native mission. Same CPUs 0–7,16–23 and software EGL. Frozen controls unchanged.

Assignment 2 completed after owner exit and persistence: no arrivals, budget
exhausted at 480.92 simulated seconds, zero contacts, 1,186/1,200 plans on time.
Physical and actual selector/model/dispatch evaluations completed. The current
reserve diagnostic found 1,090 consecutive blocked hold plans, frames 444–4800;
1,077 were on time. First/last stored clearance was 0.470600/0.454639 m;
physical body-centre-to-wall clearance was 0.478194/0.460443 m. Both were below
the 0.48-m movement threshold. No translation-veto recovery occurred.
The failure is inability to escape a current reserve deficit, not evidence
that learned forecasts are necessary. Map-discrepancy cause and safe turn-only
escape remain unisolated. No raw-depth replay is pending; preserve all
non-depth diagnosis and outcome records. Path 3.588 m, final goal distance
2.468 m. Next: the established reactive feedback control on layout 0.

Completed instantaneous and diagnosed reserved-off layout-0 depth retired
under standing policy; all non-depth comparison inputs retained. Assignment 3,
reactive_feedback/layout 0, launched in session 66182, owner PID 3952576.
Native launch and live owner confirmed. Preflight: 0.4% CPU busy,
69.06 GB available RAM, 6.22 GB artifact space; GPUs idle, no competing
native mission. Same CPUs 0–7,16–23 and software EGL. Controls unchanged.

Assignment 3 completed after owner exit and full persistence: reactive_feedback
layout 0 exhausted 480.86 simulated seconds without verified arrival; zero
contacts. Physical and selector/model/dispatch checks completed: the actual
selector used no forecast values. Plans: 1,170/1,195 on time. Terminal mode
began at frame 832 and selected 973 right turns, one forward pulse and 14
holds across 988 plans (977 on time). From that observation onward, requested
commands included 383.50 s of turns, 0.40 s of translation including queued
commands, and 13.38 s zero. Physical heading changed by -28.19 revolutions;
median physical goal distance was 35.683 mm. Observed dwell reached at most
three quiet intervals, below the required ten. This is a terminal-feedback
failure, not a deadline-dominated pause or proof of learned-model necessity.
The full diagnosis is `reactive_terminal_turn_diagnosis_v1.json`. No raw-depth
replay pending; all non-depth records retained for the comparison.

## First layout: all three controls complete

| Controller | Verified round trip | Simulated seconds | Contacts |
| --- | --- | ---: | ---: |
| Full supervised reference | Yes | 160.92 | 0 |
| Instantaneous ranking, predictive checks | Yes | 198.86 | 0 |
| Current reserve, prediction off | No arrival | 480.92 | 0 |
| Reactive heading feedback, prediction off | No verified arrival | 480.86 | 0 |

Comparison: `go2_persistent_visual_baselines_comparison_layout00_v1_attempt_001`.
The main ranking ablation succeeded; both off failures are specific controller
limitations. These outcomes do not isolate learned prediction or establish
JEPA superiority. Nine fixed assignments remain, starting with reserved_off
on layout 1. Preserve the controller and all failures throughout the batch.

Layout-0 comparison PNG/SVG saved; PNG inspected. All completed layout-0
baseline depth retired after evaluations and diagnoses, all non-depth inputs
and failed outcomes preserved. Assignment 4, reserved_off/layout 1, launched
in session 3430, owner PID 3954290. Native launch and live owner confirmed.
Preflight: 0.3% CPU busy, 68.89 GB available RAM, 5.65 GB
artifact space; GPUs idle, no competing native mission. CPUs 8–15,24–31,
software EGL. Frozen controller/plan unchanged.

Assignment 4 completed: owner absent and session terminal/missing; complete
persistence was consumed successfully by the physical/selector/model/dispatch
evaluation. Reserved-off/layout 1 exhausted 480.90 simulated seconds without
arrival, zero contacts, 1,187/1,200 plans on time. The exact current-reserve
diagnostic found 1,065 consecutive blocked holds, frames 544–4800. First/last
stored clearance was 0.472775/0.448317 m; physical body-centre wall clearance
was 0.483574/0.460242 m against the 0.48-m movement threshold. Thus the first
stored reserve rejection occurred while physical clearance still exceeded
the threshold. No translation-veto recovery occurred; last translation request
was at 56.18 s on the recorded clock. The stored-map discrepancy cause and
safe escape remain unisolated. This controller cannot escape its current
reserve deficit; the result does not establish learned forecasting necessity.
Diagnosis: `current_action_reserve_diagnostic_v1.json`. No raw-depth replay is
pending; retain all non-depth records and this failed outcome. Next fixed
assignment: reactive_feedback/layout 1, then instantaneous/layout 1.

Completed diagnosed reserved-off/layout-1 depth retired: 9,610 leaves,
2,119,110,656 allocated bytes reclaimed; all 9,655 non-depth file identities
and 36 JSON hashes preserved. Assignment 5, reactive_feedback/layout 1,
launched in session 97682, owner PID 3956303; native launch confirmed.
Preflight: 0.8% CPU busy, 68.82 GB available RAM, 5.12 GB artifact space;
GPUs idle and no competing native mission. Same CPUs 8–15,24–31 and
software EGL. Frozen controls unchanged; broader environment tests deferred.

Assignment 5 completed, owner exited zero and recordings fully persisted.
Reactive_feedback/layout 1 passed physical goal and home arrivals, actual
forecast-independent selector, model and dispatch checks: 188.46 simulated
seconds, zero contacts, 421/464 plans on time. Goal/home arrival frames
1185/1881; maximum physical distance during each one-second dwell was
18.647/7.113 mm, maximum 100-ms speed 13.875/24.642 mm/s, and every requested
command during dwell was zero. Accepted dispatch age reached 240 ms; simulation
lag reached 55.873 s, so this remains measured simulation, not real-time
qualification. This is direct evidence that the current perception/memory
package can navigate layout 1 without forecast-based action selection; the
reactive arm is now one verified round trip and one terminal-turning failure
across two exposed layouts. No learned-prediction necessity follows from the
reserve-only failures. No raw replay pending. Next: assignment 6,
instantaneous ranking/layout 1 with predictive guards retained.

Completed reactive/layout-1 depth retired: 3,766 leaves, 1,147,236,352
allocated bytes reclaimed; all 3,810 non-depth identities and 35 JSON hashes
preserved. Assignment 6, instantaneous/layout 1, launched in session 94228,
owner PID 3957381; native launch and live owner confirmed. Preflight: 0.3%
CPU busy, 68.95 GB RAM available, 4.87 GB artifact space, GPUs idle and no
competing native mission. Same odd-layout CPUs and software EGL; frozen
controller and model unchanged.

Assignment 6 completed: owner exited zero after full persistence. Instantaneous
ranking/layout 1 passed both physical arrivals and actual selector/model/dispatch
checks: 255.60 simulated seconds, zero contacts, 475/625 plans on time. Goal/home
frames 1704/2552; maximum physical dwell distances 24.231/4.694 mm and maximum
100-ms speeds 32.018/11.703 mm/s, all requested dwell commands zero. Path
16.125 m; final home distance 4.231 mm. Forecast and instantaneous main
preferences differed on 178/625 plans; retained predictive checks overrode
the instantaneous preference on nine plans. This remains a main-ranking
ablation, not a prediction-free controller. No raw replay pending.

## Second layout: all three controls complete

| Controller | Verified round trip | Simulated seconds | Contacts |
| --- | --- | ---: | ---: |
| Full supervised reference | Yes | 204.88 | 0 |
| Instantaneous ranking, predictive checks | Yes | 255.60 | 0 |
| Current reserve, prediction off | No arrival | 480.90 | 0 |
| Reactive heading feedback, prediction off | Yes | 188.46 | 0 |

Comparison: `go2_persistent_visual_baselines_comparison_layout01_v1_attempt_001`.
Full summaries and the reserve failure are retained; PNG/SVG saved and PNG
inspected. Across the first two layouts, instantaneous ranking is 2/2 round
trips, reactive feedback 1/2 and reserve-only feedback 0/2; all zero contacts.
The full supervised reference passed both. These controller packages and
variable deadline outcomes do not isolate JEPA or establish reliable speed
differences. Six assignments remain: reactive/instantaneous/reserved on
layout 2, then instantaneous/reserved/reactive on layout 3. Broader environment
tests remain deferred. No frozen controller changes during this batch.

Completed instantaneous/layout-1 depth retired: 5,108 leaves, 1,559,965,696
allocated bytes reclaimed; all 5,153 non-depth identities and 36 JSON hashes
preserved. Assignment 7, reactive_feedback/layout 2, launched in session 25794,
owner PID 3958550; native launch and live owner confirmed. Preflight: 0.2%
CPU busy, 68.96 GB available RAM, 4.52 GB artifact space, GPUs idle and no
competing native mission. CPUs 0–7,16–23, software EGL. Frozen controls unchanged.

Assignment 7 completed with owner exit zero and full persistence. Reactive
feedback/layout 2 passed physical goal and home arrivals and actual
selector/model/dispatch checks: 186.38 simulated seconds, zero contacts,
435/455 plans on time. Goal/home frames 1100/1860; maximum physical distance
during each one-second dwell was 7.706/12.191 mm and maximum 100-ms speed
19.792/12.810 mm/s, with all requested dwell commands zero. Maximum simulator
lag was 50.961 s; this is not real-time qualification. Forecast-free feedback
now has two round trips and one terminal-turning failure across its first
three exposed layouts. No raw sensor replay pending. Next fixed assignment:
instantaneous ranking/layout 2, with predictive checks retained.

Completed reactive/layout-2 depth retired: 3,724 leaves, 1,111,363,584 allocated
bytes reclaimed; all 3,768 non-depth identities and 35 JSON hashes preserved.
Free artifact space was then 4,281,479,168 bytes, just below 4 GiB. Cleared
130 pip HTTP download-cache files (about 1.528 GB) with `python3 -m pip cache
purge`, restoring 5,806,579,712 bytes free; installed packages and experiment
artifacts remain intact. Original cadenced tracking failure and current JEPA
tracking-loss sensors remain retained; their depth was not retired.

Assignment 8, instantaneous/layout 2, launched in session 58035, owner PID
3959996; native launch and live owner confirmed. Preflight: 0.3% CPU busy,
68.80 GB available RAM, 5.81 GB artifact space, GPUs idle, no competing native
mission. CPUs 0–7,16–23, software EGL. Frozen controller/model unchanged.

Assignment 8 completed, owner exited zero after full persistence. Instantaneous
ranking/layout 2 passed physical goal and home arrivals and actual
selector/model/dispatch checks: 199.98 simulated seconds, zero contacts,
431/492 plans on time. Goal/home frames 1336/1996; maximum physical dwell
distances 13.219/7.694 mm and maximum 100-ms speeds 8.812/25.208 mm/s; all
requested dwell commands zero. Path 16.153 m, final home distance 3.758 mm.
Forecast and instantaneous main preferences differed on 124/492 plans;
retained predictive checks overrode the instantaneous preference three times.
The main-ranking ablation has now passed 3/3 round trips, with predictive
guards retained. This does not establish full prediction independence or a
JEPA advantage. Maximum simulator lag 55.602 s. No raw sensor replay pending.
Next fixed assignment: reserved_off/layout 2, then the three layout-3 controls.

Completed instantaneous/layout-2 depth retired: 3,996 leaves, 1,228,275,712
allocated bytes reclaimed; all 4,041 non-depth identities and 36 JSON hashes
preserved. Assignment 9, reserved_off/layout 2, launched in session 13513,
owner PID 3961005; native launch and live owner confirmed. Preflight: 0.2%
CPU busy, 68.75 GB available RAM, 5.54 GB artifact space, GPUs idle, no
competing native mission. CPUs 0–7,16–23, software EGL. Frozen controls unchanged.

Assignment 9 completed: owner exited zero after full persistence. Reserved-off
layout 2 exhausted 480.94 simulated seconds with no arrivals and zero contacts;
physical/selector/model/dispatch checks completed, 1,176/1,200 plans on time.
There were 1,022 reserve-blocked holds (1,002 on time). First rejection was
frame 692, followed by six clear movement plans; frames 720–4800 then formed
1,021 consecutive blocked holds (1,001 on time). First/last rejected stored
clearance was 0.474584/0.467012 m; physical body-centre wall clearance was
0.484426/0.473372 m against the 0.48-m movement threshold. Last translation
request was at 73.78 s on the recorded clock; no translation-veto recovery.
Path 5.051 m, final goal distance 1.658 m. Both reserve diagnostic and detailed
stall interval are saved. Map-error cause and safe turn-only escape remain
unisolated; no raw replay pending. This is a reserve-rule failure, not evidence
that learned prediction is necessary.

## Third layout: all three controls complete

| Controller | Verified round trip | Simulated seconds | Contacts |
| --- | --- | ---: | ---: |
| Full supervised reference | Yes | 148.42 | 0 |
| Instantaneous ranking, predictive checks | Yes | 199.98 | 0 |
| Current reserve, prediction off | No arrival | 480.94 | 0 |
| Reactive heading feedback, prediction off | Yes | 186.38 | 0 |

Comparison: `go2_persistent_visual_baselines_comparison_layout02_v1_attempt_001`.
All summaries, failures and ranking/clearance diagnostics are retained. Across
three layouts, instantaneous is 3/3 round trips, reactive 2/3 and reserved-off
0/3, all with zero contacts. Full supervised passed all three. Timing and
controller-package differences prevent a clean learned-prediction attribution.
Three assignments remain on layout 3: instantaneous, reserved-off, reactive.

Layout-2 comparison PNG/SVG saved; PNG inspected.

Completed reserve-only/layout-2 depth retired: 9,610 leaves, 3,187,372,032
allocated bytes reclaimed; all 9,656 non-depth identities and 37 JSON hashes
preserved. Assignment 10, instantaneous/layout 3, launched in session 8464,
owner PID 3963000; native launch and live owner confirmed. Preflight: 0.3%
CPU busy, 68.86 GB available RAM, 4.83 GB artifact space, GPUs idle, no
competing native mission. CPUs 8–15,24–31, software EGL. Frozen controls unchanged.

Assignment 10 completed, owner exited zero after full persistence. Instantaneous
ranking/layout 3 passed both physical arrivals and actual selector/model/dispatch
checks: 244.82 simulated seconds, zero contacts, 440/594 plans on time. Goal/home
frames 1403/2444; maximum physical dwell distances 14.098/21.536 mm, maximum
100-ms speeds 16.377/17.319 mm/s, all requested dwell commands zero. Path
17.383 m, final home distance 21.667 mm. Main forecast/instantaneous preferences
differed on 206/594 plans; predictive checks overrode instantaneous preference
eight times. Maximum simulator lag 75.206 s; no real-time qualification.

The instantaneous main-ranking arm is complete: four verified round trips,
zero contacts. Across 2,199 plans, main preferences differed 678 times and
retained predictive checks overrode instantaneous preference 24 times. Both
this arm and full supervised passed all four layouts. Full supervised used
less simulated time in every recorded pair (160.92/204.88/148.42/201.68 s
versus 198.86/255.60/199.98/244.82 s). This is an observed efficiency pattern,
not a replicated performance guarantee; actual planning deadlines differed.
The arm retains predictive guards and does not establish full prediction
independence or a JEPA advantage. No raw replay pending. Next: reserved-off
layout 3, then reactive feedback/layout 3 to complete the fixed batch.

Completed instantaneous/layout-3 depth retired: 4,892 leaves, 1,458,515,968
allocated bytes reclaimed; all 4,937 non-depth identities and 36 JSON hashes
preserved. Assignment 11, reserved-off/layout 3, launched in session 86682,
owner PID 3964129; native launch and live owner confirmed. Preflight: 0.4%
CPU busy, 68.89 GB available RAM, 4.50 GB artifact space, GPUs idle, no
competing native mission. CPUs 8–15,24–31, software EGL. Frozen controls unchanged.

Assignment 11 completed: owner exited zero after complete persistence.
Reserved-off/layout 3 exhausted 480.90 simulated seconds, no arrivals, zero
contacts; physical/selector/model/dispatch checks completed, 1,185/1,200 plans
on time. The clearance diagnostic found 1,053 consecutive blocked hold plans,
frames 592–4800. First/last stored clearance was 0.478332/0.462577 m versus
physical body-centre wall clearance 0.494015/0.479540 m and movement threshold
0.48 m. Last translation request at 60.98 s on the recorded clock; no
translation-veto recovery. No raw replay pending. Map discrepancy cause and
safe turn-only escape remain unisolated; this is not learned-model necessity.
The reserve-only arm is complete: 0/4 goals or round trips, zero contacts, all
four failures retained and diagnosed. Reactive/layout 3 is the last assignment.

To retain recording headroom, reviewed and retired the completed older
persistent-routing layout-0 success's depth, ending its former reference pin.
The eight-run memory comparison and physical/scope evaluations remain complete;
no active depth replay or fit requires that root. Retired 6,212 regular single-link
depth leaves, reclaimed 1,862,594,560 allocated bytes; all 6,307 non-depth
identities and 72 JSON hashes preserved. Current tracking-loss and contact
failure sensors remain full. See the retention policy and exact inventory
`.generated/depth_retirement_completed_routing_memory_reference00_2026-09-16/`.

Completed reserve-only/layout-3 depth retired: 9,610 leaves, 2,671,222,784
allocated bytes reclaimed; all 9,655 non-depth identities and 36 JSON hashes
preserved. Assignment 12, reactive_feedback/layout 3, launched in session 30763,
owner PID 3965870; native launch and live owner confirmed. Preflight: 0.4%
CPU busy, 68.95 GB available RAM, 5.68 GB artifact space, GPUs idle, no
competing native mission. CPUs 8–15,24–31, software EGL. Frozen controls unchanged.

Assignment 12 completed: owner exited zero after full persistence. Reactive
feedback/layout 3 passed physical goal and home arrivals and actual
selector/model/dispatch checks: 186.78 simulated seconds, zero contacts,
407/459 plans on time. Goal/home frames 1095/1864; maximum physical dwell
distances 15.870/17.164 mm, maximum 100-ms speeds 27.464/28.695 mm/s, all
requested dwell commands zero. Maximum simulator lag 55.833 s. No raw replay
pending. The reactive arm is complete: 3/4 round trips, zero contacts.

## Fixed baseline batch complete

All twelve assignments are physically evaluated, with all original outcomes
retained and source hashes still matching the frozen plan. Instantaneous
ranking passed 4/4 round trips, reactive feedback 3/4 and reserve-only feedback
0/4. All recorded zero contacts. The layout-3 comparison PNG/SVG was saved
and the PNG inspected, completing all four layout comparisons.

Aggregate `go2_persistent_visual_baselines_complete_v1_attempt_001/result.json`
also includes the preceding sixteen model-comparison missions: 21/28 round
trips, 22/28 outbound arrivals, zero contacts. All 28 recorded same-window XY
comparisons favour pose/command over neural forecasts. These are four layouts,
one execution per condition/layout, with substantial timing and controller
limitations; no JEPA advantage or general reliability claim is established.
Full interpretation and next work are in
`docs/go2_persistent_visual_baselines_complete_result_2026-09-16.md`.
The broad goal remains incomplete; broader environment tests remain deferred.

Final reactive/layout-3 depth retired after evaluation: 3,732 leaves,
1,137,950,720 allocated bytes reclaimed; all 3,776 non-depth identities and
35 JSON hashes preserved. Artifact space afterward: 5,433,380,864 bytes.
All twelve baseline roots retain complete non-depth evidence and explicit
depth-retirement markers. No native mission is running from this batch.
