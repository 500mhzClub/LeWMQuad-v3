# Geometry-progress pilot: informative actions, failed sensing gate

The fresh 24-episode pilot establishes the intended **native action/progress
reversal**, but **does not pass the combined measurement gate**. The original
audit and a separate command-representation readout remain failed. A complete
available-evidence readout retains all episodes and failures. An additional
physical-visibility check found eight frames where the inherited 5cm render
near plane clips an opaque wall, including four future-image target frames.
No training, learned-policy execution, maze qualification or overall-goal
completion occurred.

This advances handoff section 10B: geometry now changes the action that makes
contact-free progress, and holding/turning alone cannot win the progress task.
The next step is a prospective acquisition successor using the repository's
existing 5mm core-ordered camera and modality-specific validity checks. The
choice of the older acquisition stack in this pilot was a regression relative
to the later independent RGB/body collection; its sensing implementation must
not be scaled unchanged.

The durable joined numbers are in
[the compact scientific readout](go2_geometry_progress_scientific_readout_2026-09-07.json).
The original handoff and its eleven recorded file hashes are unchanged.

## Fixed experiment and actual outcomes

Two mirrored partial panels × two appearance seeds × six actions. Episodes
were assigned opaque names in a fixed seeded shuffle. This is four development
geometry/appearance strata and zero independent maze evaluation layouts.
Nominal support, existing checkpoint gait, CPU simulation, hidden robot and ideal
body sensors. Three quiet 100ms ticks establish the existing four-packet history
after 1.5s settling. Each candidate declares a 40-tick horizon, including ten
terminal zero ticks. Commands are bounded by vx<=0.20m/s and |yaw|<=0.45rad/s.

The fixed evaluator goal is (1.2,0)m in the actual departure body frame.
Successful local progress requires a complete horizon, no contact or other stop,
and at least 0.15m actual reduction in goal distance. The goal has not been
reached, and the robot has not yet traversed the passage at the end of these
prefixes. Displacement comes from recorded native poses, not command duration.

| Action | Full horizons / 4 | Successful progress / 4 | Contact stops / 4 | Complete-horizon progress |
| --- | ---: | ---: | ---: | ---: |
| Hold | 4 | 0 | 0 | 0.69mm |
| Forward | 0 | 0 | 4 | unavailable; stopped prefixes reduced distance by 271.74mm |
| Left arc | 2 | 2 | 2 | 313.25mm in left-open scenes |
| Right arc | 2 | 2 | 2 | 351.08mm in right-open scenes |
| Left pure turn | 4 | 0 | 0 | 8.57mm |
| Right pure turn | 4 | 0 | 0 | 48.82mm |

Both appearance seeds have the required mirrored arc-success reversal. All
hold/pure-turn controls fail progress. No constant action succeeds across all
four strata. The native action-design gate passes. Four successful candidate
episodes out of 24 is not an evaluated navigation-policy success rate: the
cohort deliberately includes unsafe and nonprogress controls.

All 24 initial model contexts materialized through the existing tensor contract:
RGB [4,3,96,128], body [4,20,63], control [4,15,7], candidate actions [8,5,3].
There are four distinct RGB histories but only one body history and one control
history. All 20 nonreference within-stratum complete sensor-prefix comparisons
are exactly equal in this cohort. The prospective independent-episode design
remains explicit; this does not fix or supersede the earlier room-return RGB
pairing failure. The new data establishes task informativeness, not learned RGB
or JEPA benefit.

[Native base-centre path figure](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_geometry_progress_science_v1_attempt_001/native_paths.png)
([SVG](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_geometry_progress_science_v1_attempt_001/native_paths.svg)).
The figure shows base centres; crosses are contact-stop endpoints, not contact
locations or a swept-volume representation.

## Complete raw accounting and preserved failures

Collection completed with exit 0: 24 episodes, 16 full schedules, eight contact
stops, 906 camera/decision packets and 62,194 native 2ms samples. All 3,510
expected artifacts are present; their pre-terminal total is 1,705,235,792 bytes.
All eight physical stops remain in the cohort, including their partial ticks.

1. The original frozen audit failed on episode 000 before producing a completed
   episode receipt. It cast requested commands to float32 before exact comparison
   with the recorder's float64 requests. In this collector, 0.45 remains a
   float64 request; actuator clipping operates on float32 and records that result
   promoted to float64. The failed comparator expected 0.44999998807907104 as the
   original request. Original audit status remains `TERMINAL_RAW_AUDIT_FAILURE`.
2. A separate command-only diagnostic checked all 24 traces. Exact float64
   request identity and exact float32-clipped applied-command identity passed,
   including both recorded applied-command fields. New tests reject one ULP of
   request or actuator corruption. The separate raw readout retained all other
   checks, completed episodes 000--007, then failed on insufficient depth rays
   in episode 008. That readout remains failed too.
3. The separately frozen available-evidence readout completed all 24 episodes.
   It retained every original comparison threshold, changed the insufficient-ray
   diagnostic from early termination into an explicit failed-frame record, and
   replayed the remaining raw sensor, command, setup, contact and stop checks.
   Exact packet reconstruction is distinct from physical measurement validity.

The available-evidence readout reconstructed all 906 packets. Its inherited
renderer-relative depth comparator passed 904 frames; the maximum error among
compared rays was 33.72 micrometres. Episodes 008 and 021, both right-open forward
actions, have zero eligible depth rays and zero public-depth-valid pixels at
frame 16 (3.1s). These are failed checks with null error, never zero-error passes.
The combined design-and-measurement gate is false.

Target accounting retains all 192 half-second horizon slots: 160 native motion
targets, 160 future-image packets, 192 known contact labels and 32 positive
contact labels. All 32 positives have missing future images and missing terminal
motion. No post-stop future was synthesized. Materialization demonstrates schema
compatibility only; it does not grant training eligibility.

## Physical occlusion: additional failure beyond the zero-ray frames

Inspection of authenticated camera/geometry/depth artifacts identifies the
5cm near-plane issue. In both zero-ray episodes, the camera is outside the panel:
at frame 15 its x coordinate is 0.53784m, at frame 16 0.55666m, and at frame 17
0.57448m; the panel's front face is x=0.61m. At frame 16, expected wall depths
are 0.053013--0.053684m, below the comparator's 0.055m inclusion margin but above
the render near plane. At frame 17 the nearer opaque wall is clipped. The
renderer-relative reference also excludes that wall, so passing that reference
does not establish physical occlusion fidelity.

The existing, unchanged `lewm/physical_first_surface_depth_development.py`
evaluator was then applied to **all 906 frames**, independently of render
clipping. Its separate diagnostic completed with these results:

- Eight failed physical-visibility frames: frames 17 and 18 of forward episodes
  006, 008, 014 and 021.
- 37,242 sampled rays with clipped opaque surfaces; 35,792 falsely public-valid
  near-surface rays. These are ray counts, not independent trials.
- Zero failed initial-history frames, but **four failed future-target frames**
  (frame 18 in each forward episode).
- This is a sampled static-opaque-surface check, not full-image/self-occlusion or
  hardware qualification. All original pixels and labels remain unchanged.

Thus even the 160 materialized future-image packets are not all physically
eligible JEPA targets. A future data contract must keep availability, physical
visibility and native motion/contact validity separate. The current scientific
readout is preserved, and this additional finding is joined explicitly in the
compact readout instead of rewriting the frozen result.

## Implementation, tests and resource limits

New core: `lewm/geometry_progress_pilot_development.py`; adapter:
`lewm/geometry_progress_learning_sample_development.py`. The collector uses
`scripts/geometry_progress_physical_init_development.py` and
`scripts/geometry_progress_session_development.py`. Its runner is
`scripts/run_go2_geometry_progress_pilot_v1.py` and frozen original auditor is
`scripts/audit_go2_geometry_progress_pilot_v1.py`.

Separate readouts are `scripts/read_go2_geometry_progress_commands_v1.py`,
`scripts/read_go2_geometry_progress_available_evidence_v1.py`,
`scripts/geometry_progress_available_sensor_evidence_development.py`,
`scripts/read_go2_geometry_progress_science_v1.py` and
`scripts/read_go2_geometry_progress_physical_visibility_v1.py`.

52 distinct focused tests passed: pilot 22; tensor adapter 6; science reader 6;
command representation 11; available-evidence semantics 7. Tests cover causal
packet rejection, fixed balanced plans, actual serialization, stop/censoring,
full-denominator scoring, isolated source changes, empty-ray failures, and an
untrained forward pass through the existing cumulative JEPA architecture.
No optimizer steps or learned checkpoint selection occurred.

Preflight: 16 physical / 32 logical CPUs, 81,683,972,096 bytes available RAM and
106,517,790,720 free artifact bytes. Serial CPU scene, one OpenCV/BLAS thread,
3GiB planned storage and 40GiB reserve. No heavy concurrent job during collection.
Median observation/control time was 126.58ms; 833/906 decisions exceeded 100ms,
before physics-tick time. This includes the shadow observer and synchronous
acquisition/recording. Physics paused during computation; real-time and realistic
hardware sensing remain unresolved.

## Artifact identities and terminal state

All roots below are children of
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1`.
Every launched process is terminal. Do not rerun, resume or overwrite these roots.

| Root | File | SHA-256 |
| --- | --- | --- |
| `go2_geometry_progress_pilot_v1_attempt_001` | `launch.json` | `265357dc47e8ceaecf7932f63441cd7020312c14bcce182e0cd07563888a2323` |
| same | `result.json` | `067a03f208cfc389e08ebba25b2c7a739dfb1b62f2001e1dc4238140cadfe6cc` |
| same | `geometry_progress_audit_failure.json` | `28318e3a5510bcbb04009af3d78f8a2703f2fcd0bc9430b053b00588e2bca3d7` |
| `go2_geometry_progress_command_readout_v1_attempt_001` | `launch.json` | `2ae65e7d5fe774c16a2a2a7e0becc4953681a8d96ee2f54eee4c546cf2001a91` |
| same | `failure.json` | `6cbec6405ef94b34c9ab56eba276223968514fae2f32904a91be339666d67c7b` |
| `go2_geometry_progress_available_evidence_v1_attempt_001` | `launch.json` | `c7093baaf213bf52185e7f0681afc3932031d969183f33c7246ea037d190f5cf` |
| same | `result.json` | `5fb9e3f490ad2a75f95a8a04b92ae85b495d69b9b0a3d5f0eec2e28be8eabdd7` |
| `go2_geometry_progress_science_v1_attempt_001` | `result.json` | `e3255efaac6d4d759dfe6c9b5c5f44f3e3ca58c83d4184ecb132a0c94d22bc8b` |
| same | `native_paths.png` | `ce1c985bdfff186664d66ce83febf053a1e98dce34626d17b1d3f680d61ec088` |
| same | `native_paths.svg` | `487fea8cb1c041c93b0e38bc58c37265f955534c73cb9771f66f6d906d072933` |
| `go2_geometry_progress_physical_visibility_v1_attempt_001` | `result.json` | `9bb467653bc6e733c185af103180b28191723d2e57cd98db0e3377a99ecce85b` |

Collection source bindings: 731. Science reader: 743. Final physical-visibility
diagnostic: 744. Original sources, `AGENTS.md`, `.ignore`, and all eleven handoff
file bindings remain unchanged. New source and reports remain local/uncommitted.

## Ordered continuation

1. Keep this native action bank and its informative result. Freeze a separately
   named successor that uses `NearFieldCapture`, `CoreOrderedDynamicSession`
   semantics, the 5mm camera constructor and explicit ordered-raster witnesses.
   Relevant existing sources: `scripts/core_ordered_dynamic_session_development.py`,
   `scripts/ordered_dynamic_physical_init_development.py`,
   `scripts/near_field_rgbd_capture_development.py`,
   `scripts/run_go2_independent_rgb_body_collection_v1.py`,
   `scripts/independent_rgb_body_audit_development.py` and
   `lewm/raster_footprint_visibility_development.py`. Adapt the new geometry/action
   inventory explicitly; do not monkeypatch or invoke a frozen old cohort.
2. Use the exact command-representation validator from this turn. Include native
   first-surface and raster-footprint eligibility before launch, with independent
   RGB/body/future/contact masks. Public depth may honestly be unavailable near
   a wall; opacity must still be preserved. Do not lower a failed threshold or
   treat a clipped foreground as free space.
3. Run a bounded fresh sensor/action pilot, retain every failure, and require the
   complete design plus measurement contract before a larger collection. Then
   build genuinely independent geometry/layout splits and matched full-RGB,
   no-RGB, action/time, direct, supervised-rollout and JEPA comparisons. Do not
   repeat the old unchanged 36-fit study or fit a winner on these exposed strata.
4. Use the learned model online to select and execute progressing actions, then
   integrate continuous joint tracking, memory, actual backtracking and novel-maze
   goal reach. Profile and account for real sensing/control latency. The earlier
   unpaired 2/3 room-return result remains a component result. The long-term goal
   remains active; learned closed-loop maze navigation is still unverified.
