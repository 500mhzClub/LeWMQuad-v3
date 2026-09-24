# Frontier retirement after a distant view

The completed contact-score pilot's disabled layout 0 failed outbound despite
4,805 accepted poses. Its full recording is retained at
`go2_contact_score_ablation_pose_command_xy_disabled_noise_2mm_native_layout00_4800_v1_attempt_001`.

`scripts/reconstruct_go2_frontier_stall_development.py` reconstructed all 1,202
actual mapping updates in recorded completion order, using the delivered noisy
camera packets and saved registered poses. Both delivered depth hashes are
checked by `NoisyPublicReplay`; no native pose or maze geometry enters the map.
The registered estimator is not rerun. All 685 selected planning records match
the reconstructed floor/fine-obstacle counts. All nine view-completion exclusion
counts and the final exact 58-cell exclusion set match. Relevant recorded route
statuses also match. Reconstruction took 124.68 seconds. Matching counts alone
do not prove every map cell identical; no original full-map snapshots exist.

The run's `frontier_map_reconstruction_v1` contains snapshots, 21 route-query
probes, and `result.json`. `scripts/analyze_go2_frontier_exclusions_development.py`
produces `exclusion_diagnosis.json` and a visually inspected PNG/SVG figure.

At planning frame 1204, the two upper-branch view completions have excluded
targets near (0.475, 2.575) and (0.425, 2.475) m. These views ended respectively
about 0.314 m and 0.224 m from their targets. The map contains observed,
coarse-traversable upper-passage cells beyond a small unobserved floor gap, but
they are disconnected from the robot's coarse traversable component. The
controller chooses the distant lower branch. Removing only the upper exclusions
from this same map restores a route to (0.475, 2.575).

At the last selected plan, frame 3736, all eleven frontiers in the 1,039-cell
robot-connected traversable component are excluded. Removing only the upper
exclusions again restores a route back to that frontier. The original policy
excludes nearby floor targets after completing a distant view, even when the
view has not resolved the coverage gap. Its later exhausted scan budget explains
the absence of further selected plans. This establishes a route-selection
mechanism in the saved failure, not a successful alternative execution. The
physical cause of the small unobserved floor gap is not yet isolated.

## Focused prospective treatment

`ArrivalConditionedFrontierVisits` preserves the initial standoff view. If that
view finishes more than the existing 0.10 m frontier-arrival radius from its
target, it defers frontier exclusion. A subsequent selection of those target
cells follows the existing observed route until the original arrival trigger
requests another view. Exclusion occurs only after a completed view within
0.10 m in the registered pose. This is not a new physical-arrival certification.
No unknown floor is filled and no command check is relaxed; all downstream action,
obstacle, speed and timing checks remain. The existing nearby-panorama directed
view rule remains active. A focused regression reproduces premature retirement
in the original strategy and checks approach followed by a closer completed view.

Before dispatch, fix two new development probes on the already exposed layout 0:
learned and disabled contact scores, both with this frontier treatment. Retain
the frozen supervised model, pose/command XY, learned yaw, 2 mm delivered noise,
mapping, actions, recovery, arrival rules and 4,800-tick budget from the pilot.
This diagnoses the shared exploration mechanism; it does not expand or replace
the completed pilot. Both new outcomes count, with no tuning between them.
There is no fresh original-policy pair, so comparisons to the preceding pilot
are development evidence with asynchronous trajectory variation.

Launcher: `scripts/run_go2_arrival_conditioned_frontier_development.py`, argument
`--contact-score learned|disabled`. Learned runs on CPUs 0–7,16–23 and disabled
on 8–15,24–31, at most two native owners including archiving. About 11 GiB of
artifact storage is available; two maximum-length recordings fit the recent
observed envelope. Preserve both full recordings for evaluation and diagnosis.
Root template:
`go2_arrival_conditioned_frontier_contact_<mode>_noise_2mm_native_layout00_4800_v1_attempt_001`.

Independently evaluate physical goal/return, contacts and failures after each
owner exits. Check whether the upper gap becomes connected and report deferred
views and later target approaches. Neither replay queries nor the regression
test prove the treatment succeeds. No native outcome is known at this entry.

The focused regression passed (one test, 1.61 s). The first collection command
omitted the repository's required `lewm_genesis:lewm_worlds` import paths and
failed before executing a test; the corrected normal environment passed.

## Dispatch update

The learned-contact probe launched in session 17079, owner 3610524, and its
actual launch record confirms the treatment, supervised model, pose/command XY,
learned contact mode and measured-simulation clock. Native progress reached
camera frame 200 with continued outbound tracking at this entry.

The disabled launch, session 60123, exited 1 after 6.05 s before native execution.
The inherited transfer launch writer requires layout 0 on CPU group 0, despite
the new outer launcher's group-1 assignment for this mode. Its created output
root contains an empty native directory and no launch record; `startup_failure.json`
preserves this failure. No scientific outcome occurred and it is not counted as
a navigation failure or replaced silently. Run its native probe sequentially on
CPU group 0 after the learned owner exits, using a new attempt-002 output root.
The treatment/model/parameters remain fixed; only launch scheduling/root identity
needs correction. Thus only the learned-contact simulation is currently live.

## First physical outcome verified

The learned-contact owner exited 0 after 497.53 s including archive, peak RSS
19,494,212 KiB, zero swaps. All 3,545 camera observations have registered poses.
Independent physical evaluation passes goal frame 1958 and return frame 3543,
with zero disallowed contacts. Maximum dwell distances are 22.542/18.711 mm,
maximum 100-ms dwell speeds 0.018895/0.017391 m/s, and all dwell requests zero.
Median/max position error is 2.753/5.625 mm; final home distance 18.862 mm.
The 863 fully executed 700-ms windows have applied XY RMSE 10.442 mm,
maximum 34.632 mm, and ten path errors above 30 mm. These errors do not certify
clearance or speed.

One frontier view completed 309.963 mm from its target; exclusion of nine cells
was deferred. There were no later frontier-view completions and no exhausted
view-budget records; pipeline faults are empty. The resulting navigation succeeds,
but this single enabled-contact probe does not show the change was necessary.
The disabled-contact probe remains the more direct test of the saved failure.

After the learned owner exited, the launcher was changed only to require CPU
group 0 for both modes and to use attempt 002 for disabled mode, preserving its
failed startup. The runtime treatment class, model and scientific parameters
were unchanged. A memory-only launch-writer check exercised the full inherited
writer on the required affinity before dispatch. About 8.5 GiB of storage remains.

Disabled attempt 002 then launched in session 94033, owner 3612290, confirmed
live. Its actual launch record matches disabled contact, pose/command XY,
arrival-conditioned frontier retirement and the two-probe layout-0 scope.
Its navigation and independent physical outcome are pending.

## Disabled probe failed in the new closer-view integration

Disabled attempt 002 exited 1 after 154.54 s including archive, peak RSS
7,322,928 KiB, zero swaps. It acquired 994 camera pairs and published 993 poses,
with no arrivals and zero disallowed contacts. Median/max position error is
1.378/5.162 mm; final goal distance 3.257 m. Its 237 fully executed 700-ms
windows have XY RMSE 4.966 mm, maximum 12.536 mm and no path error above 30 mm.
The full sensor recording, failure and independent evaluation remain retained.

The new treatment deferred exclusion after views completed 224.704 and
177.171 mm from their targets. It then approached the latter target and
completed a nine-stage panorama 35.863 mm away, excluding seven cells. The
next nearby-view query raised `KeyError('view_start_map_xy_m')`. The base
non-standoff visit records no start position, while nearby-panorama selection
expects this field on completed panoramas. Thus this native run does not resolve
the original exploration hypothesis; it exposes an integration bug in the new
arrival-triggered path. There is no perception-failure attribution.

The completed two-probe result, including this failure, is preserved at
`go2_arrival_conditioned_frontier_layout00_summary_v1_attempt_001/result.json`.
Its four-panel PNG/SVG comparison with the preceding pilot was visually checked.
Actual motion/contact channels, scored contact probabilities, shared settings
and common predecessor runtime sources agree; the new probes differ in launcher
hash only because of the documented sequential scheduling/startup recovery.

## Fixed single follow-up after the metadata repair

The regression `test_completed_arrival_panorama_can_support_a_nearby_directed_view`
reproduced the exact native KeyError after a full nine-stage arrival panorama.
`ArrivalConditionedFrontierVisits` now records the actual start position when
every new view begins, including non-standoff views, and preserves that position
through subsequent stages. Both focused frontier tests pass (1.63 s). Existing
retirement distances, headings, panorama pruning, model, fitting, actions and
physical guards are unchanged.

Fix one follow-up before dispatch: disabled contact on the same exposed layout,
same 4,800-tick budget, with this metadata repair. Use CPU group 0 and root
`go2_arrival_conditioned_frontier_contact_disabled_noise_2mm_native_layout00_4800_v1_attempt_003`.
The same launcher now exposes only this disabled follow-up and identifies the
repair and single native assignment in the launch receipt. This is a new
development follow-up, not a replacement result within the completed two-probe
comparison. Retain and report attempt 002's native failure alongside attempt
003's outcome. No navigation success is established by the passing regression.

Attempt 003 launched in session 85939, owner 3613466, confirmed live. Its launch
record identifies the view-start repair, disabled contact, pose/command XY and
one planned disabled assignment. About 7.9 GiB was available before dispatch.

## Repaired follow-up: verified goal and return

Attempt 003 exited 0 after 517.19 s including archive, peak RSS 19,876,020 KiB,
zero swaps. All 3,625 camera observations have registered poses. Independent
physical evaluation passes goal frame 2406 and return frame 3623, with zero
disallowed contacts. Maximum dwell distances are 21.549/18.109 mm; maximum
100-ms dwell speeds are 0.000612/0.019932 m/s, with all dwell requests zero.
Median/max position error is 1.789/9.080 mm, final home distance 18.271 mm and
recorded horizontal path length 24.114 m. Its 761 fully executed 700-ms windows
have applied XY RMSE 6.908 mm, maximum 25.404 mm and no path error above 30 mm.

Seven frontier views completed: four exclusions were deferred and three views
completed within the 0.10 m arrival radius. At the upper passage, the view at
target (0.525, 2.625) m first ended 252.692 mm away, leaving its targets eligible.
The robot then approached and completed another panorama 23.795 mm from that
target. It subsequently followed the upper passage to the goal and returned.
All seven view-start fields exist and the recorded exclusion decisions match
the measured completion distances. There are no exhausted-view-budget records
or pipeline faults. Twenty-two frontier cells were ultimately excluded.

The retained report
`go2_arrival_conditioned_frontier_view_start_repair_summary_v1_attempt_001/result.json`
includes the original stall, the failed closer-view integration attempt and
this repaired success. Its three-panel PNG/SVG was visually inspected. All 893
actual selected-plan contact/motion bindings and scored zero contact terms
were checked; 134 common predecessor sources match. The two deliberate changed
sources versus attempt 002 are the frontier view-start metadata and follow-up
launcher. All prior failure outcomes and full sensor recordings remain intact.

There are 768 on-time and 125 late selected plans. Of the late plans, 115 occur
on observed goal routes; 83 fall in frames 2500–2999 during the return. Maximum
recorded observation-to-plan completion is 1,264 ms. The run's maximum host/sim
lag is 10,289.510 ms and camera acquisition maximum is 664.084 ms. This remains
measured-simulation evidence, not real-time qualification or a speed comparison.

Interpretation: the repaired closer-view policy can overcome the diagnosed
frontier stall on this exposed maze. The recorded approach/view sequence supports
the proposed mechanism, but one follow-up with asynchronous trajectory variation
does not establish repeatability, general reliability or a contact-score ranking.
The original two-probe outcome remains one success and one software failure;
the follow-up success is reported separately. The small floor-coverage gap's
sensor-level cause has not been independently isolated.

No native simulation is running after this follow-up. About 5.4 GiB of artifact
space remains. Next, keep the repaired frontier policy fixed while testing
learned versus integrated command yaw with pose/command XY and disabled contact.
That comparison will isolate the remaining neural motion channel. It needs
fresh reference executions and retention review before its four recordings;
broader new-maze and realistic timing/sensing validation remain outstanding.
