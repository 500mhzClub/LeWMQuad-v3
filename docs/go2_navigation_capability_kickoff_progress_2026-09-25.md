# Navigation capability kick-off: development progress

The authoritative programme is the [25 September brief](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md). This is a progress record, not the final capability report.

Pre-registration was committed as `bb9ab232`, before generation or physics. The 10 development, 20 validation and 60 sealed test mazes and fixed episode packets passed structural registration; initial harness/source registration was committed as `523005c1`. No validation or test episode has run or been rendered.

## First oracle development pilot

| Quantity | C0, development maze 00 / episode 0 |
|---|---:|
| Native beacon arrival | Passed, 64.8 s |
| Native home arrival | Passed, 114.8 s cumulative |
| Recorded mission duration | 114.82 simulated seconds |
| Episode wall time | 676.10 s |
| Wall seconds / simulated second | 5.89 |
| Outbound / return SPL | 0.836 / 0.914 |
| Disallowed contact samples | 0 |
| Hard / operating-margin violations | 0 / 0 |
| Unresolved sampled clearance | 0 |
| Minimum all-primitive clearance lower bound | 0.17273 m |
| Matching candidate-prefix comparisons | 1,668 passed |
| Maximum matching-prefix position error | 0 m |
| Median / p95 planning latency | 1.677 / 1.735 s |
| Hold plans, outbound / return | 2/156 / 1/122 |

Both the existing initial-frame physical arrival criteria and distance to the generated fixed world beacon/home pass. Articulated clearance was evaluated at every native 2-ms step, with all 27 collision primitives. The FK interval robustness check also has no threshold failures. These results establish one successful development episode; they do not pass the 19/20 oracle gate or establish unseen-maze capability.

The oracle uses only true candidate motion in the prediction slot. The unchanged controller receives its ordinary sensor packets, measured pose, observed map and mission instruction. Matching-prefix checks compare every native sample under the same applied tape; an override or subsequent command change terminates that matching prefix.

An executed-command coverage check confirms that all 5,460 admitted 20-ms intervals, from 276 actually admitted plans, fall inside the corresponding selected branch's verified prefix. Of 278 selected plans, 268 matched through 700 ms and nine through 800 ms. One selected turn matched only the common 300-ms prefix because dispatch vetoed it before execution. Thus the 1,668 all-candidate comparisons include common pre-dispatch prefixes; they should not be read as 1,668 executed alternative actions. See `oracle_execution_coverage.json`.

Runtime evidence is under:

`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/runs/v0_pilot_C0_dev00_ep0_attempt002`

See `episode_evaluation.json`, `oracle_prefix_check.json`, `native/physics_trace.npz`, `native_clearance_summary_arrays.npz`, `planning.json`, `requests.json` and `result.json`.

## First command-history development pilot

C1 also completed development episode 00/0 on the unchanged v0 stack. Beacon arrival was at 68.5 simulated seconds and home arrival at 111.2 seconds. The complete recording took 905.59 wall seconds (8.14 wall seconds per simulated second). Outbound/return SPL were 0.821/0.928. There were zero disallowed contacts, zero hard or operating-margin violations, and no unresolved sampled-clearance intervals; the minimum articulated separation lower bound was 0.15585 m. Hold decisions were 2/166 outbound and 2/104 on return. Median/p95 planning latency was 2.607/2.668 wall seconds.

The source RGB and depth hashes were retained successfully for this episode. The original recording occupies approximately 176 MiB. Its replay pipeline test also tests prospective omission of the neural workload that the deployed C1 selector does not use; that optimisation requires identical command-model predictions, decisions, commands and native physics records before admission. The full original neural workload remains the measured baseline above.

Evidence is in `runs/v0_pilot_C1_dev00_ep0_attempt001` under the same fresh output root. This is a development pilot, not a capability estimate or a completed ten-episode harness screen.

### Replay and throughput optimisation

The complete C1 source replay matched all 2,226 original RGB images bitwise, all 270 selected actions, all 5,561 final command steps, and every native physics field exactly. The fitted command-history predictions also matched exactly with the unused neural workload omitted. This supports that omission for C1 only; C2 requires its own check. The source recording and baseline timing remain unchanged.

The first pipeline video contains 3,337 frames at 1920×1080, 30 fps, H.264/yuv420p. Its separate chase replay had zero position and yaw error. Visual review identified frequent robot occlusion with the original low chase offset; a steeper fixed camera offset is prepared for a separate render pass, without changing simulator or controller code. The prototype is a pipeline test, not an official capability video. Its files are under `videos/pipeline_test_attempt001`.

A read-only lossless RGB archive check reproduced all 2,226 pixels-per-frame hashes, but saved only about 7.6% against the existing PNG recordings (88,378,315 versus 95,683,985 bytes). It is not adopted as a storage solution. The original images and this measured check are preserved.

Read-only final-rule accounting resolves the holds that the reused historical classifier initially left unexplained. C0 had two holds from predicted-footprint observation coverage and one intended quiet-arrival hold; C1 had three coverage holds and one intended quiet-arrival hold. Coverage uses both the observed floor and the motion forecast, so these are not observation-only exclusions. Neither successful episode supplies evidence of a sustained stall. Per-run `hold_analysis_v2.json` addenda preserve the original evaluations; the closed Stage A report is unchanged.

## Preserved implementation issues

The first oracle attempt stopped after settling, before any mission command or branch, because its new `nn.Module` retained the default training flag. The evaluation-mode correction and fresh attempt were recorded in [adapter revision 1](go2_navigation_capability_harness_v0_adapter_r1_2026-09-25.json). This changed no controller algorithm.

The successful pilot's owner called physical persistence but omitted the separate observation-persistence method. Physics, decisions, commands, observed poses and the initial snapshot survive; the original camera images and their per-frame hashes do not. Its physical arrival results use an exact timestamp join between retained acquisitions and native samples, with the unchanged dwell/speed/zero-command thresholds. This reconstructs no image evidence. This pilot cannot provide a source-image-verified video.

[Adapter revision 2](go2_navigation_capability_harness_v0_adapter_r2_2026-09-25.json) invokes the unchanged observation writer. The original pilot is preserved and is not repeated. The pipeline video is assigned to the first C1 pilot before that run begins.

C4's first startup stopped before encoder calls or optimiser updates: `torch.__version__` is a string subclass rejected by the strict converter comparison. The successor records it explicitly as a native string; the converter and fixed training plan are unchanged. The failed root is preserved and its elapsed GPU-owner accounting carries forward.

## C4 and remaining work

The first reactive C2 pilot also completed episode 00/0: beacon at 85.3 s, home at 125.2 s, and 1,009.90 wall seconds (8.07 wall seconds per simulated second). Outbound/return SPL were 0.763/0.946. It had no selected holds, disallowed contacts, hard/operating-margin violations or unresolved sampled clearances. Minimum articulated clearance lower bound was 0.11676 m. Its baseline still includes the original unused neural workload; its omission is being checked separately against all original source inputs and outputs.

That C2 omission check has now passed: 2,506 bitwise RGB matches, 304 identical selections, 6,261 identical commands and every native physics field identical. The whole checked replay took 258.08 wall seconds for 125.22 simulated seconds, including its verification overhead. The C2 result's inherited generic prediction-slot label should be read as “unchanged reactive selector, unused neural computation omitted”; no recorded learned forecast was used by this check.

The higher-camera pipeline video also passed: 3,337 frames, zero replay pose error, the required 1080p/30-fps/H.264/yuv420p format, and a contact-sheet visual review. Its directory is `videos/pipeline_test_chase_revision002`, including `metadata.json` and `visual_review.json`. It reuses the already-verified C1 decision replay and adds only a separate chase pass. Neither controller nor simulator code changed.

These are three paired development pilots on one episode, not three validation capability estimates. No harness change has been made.

The simulator's existing EGL renderer uses the integrated Radeon device (`renderD129`, PCI `7b:00.0`), whereas C4 encoding/fitting explicitly uses the discrete R9700 (`cuda:0`, PCI `03:00.0`). C4 can therefore continue during CPU-motion replay checks without competing for their rendering GPU. New neural source pilots still require the discrete GPU exclusively. Device memory reserves count all users.

All 8,414 prescribed training contexts have their causal RGB and command histories: 5,966 original/heading contexts and 2,448 maze contexts, from 156 training-role recordings. There are 9,948 unique causal image paths. No new development, validation or sealed material enters fitting. The direct model has 17,397,283 trainable parameters; a synthetic CPU check verified its candidate/horizon interface. Training-render provenance remains unverified.

The fixed C4 fit is complete: 1,760 updates, 17,397,283 parameters, 4,869.74 seconds (1.353 hours) of conservative GPU-owner accounting including recorded pauses, within the 12-hour cap. No validation data or model selection was used. Its final checkpoint SHA-256 is `d491584b17ee4321d65daf883460ccf3b626fe8d735b986473d2ecc477064013`. [Adapter revision 4](go2_navigation_capability_harness_v0_adapter_r4_2026-09-25.json), committed as `872c1e11`, binds that checkpoint and the C1/C2 equivalence results before any C4 navigation.

The C3 serial pilot completed with the pre-declared maze-data head. C4's serial pilot is complete; see its physical-arrival failure below. The two/four-owner checks run after all five serial pilots. Before C3's result, their implementation plan was refined to exercise C3's learned GPU workload if it supplies a complete recording without a source exception; otherwise they use the retained C1 trace. Mission success is not a selection condition, and no science trial is replaced. Only the tested controller type can subsequently use concurrency. This replaces the unexecuted C1-only check plan while keeping the pre-registered two/four counts and all programme caps. The full measured budget projection, harness iteration, oracle gate, validation capability results, official videos and E1 proposal remain outstanding.

## First JEPA development pilot

C3 completed episode 00/0 safely: beacon at 150.1 s and home at 238.7 s (238.72 s recorded). It required 1,937.24 wall seconds, or 8.115 wall seconds per simulated second. Outbound/return SPL were 0.5801/0.8593. Native articulated evaluation found zero disallowed contacts, zero hard or operating-margin violations, and no unresolved sampled clearances; the minimum separation lower bound was 0.12615 m. The FK interval robustness check also passed.

It selected hold on 78/370 outbound and 72/218 return decisions: 150/588 overall. Retained-rule accounting identifies 100 score losses, two motion-dependent observation-coverage overrides, and 48 planned arrival-settling holds. These counts describe one successful development trajectory, not an isolated causal readout effect or a capability estimate. Median/p95 planning latency was 2.608/2.667 wall seconds. Training-render provenance remains unverified.

Evidence is in `runs/v0_pilot_C3_dev00_ep0_attempt001`, including its unchanged source recording and `episode_evaluation.json`. No harness or learned model was changed in response.

## First supervised development pilot and reader erratum

C4 recorded 188.42 simulated seconds in 670.57 wall seconds (3.559 wall/sim). The controller declared beacon arrival at 71.5 s and home arrival at 188.4 s. Native checks confirm home arrival but reject the beacon dwell: maximum distance from the fixed generated beacon was 43.894 mm, exceeding the pre-registered 40-mm radius. The existing initial-frame target check passes; the fixed-world target check does not. Thus beacon and round-trip success are false. This is retained as an arrival-verification failure, not silently repeated or relaxed. No model or harness repair follows from this one pilot.

There were zero contacts, hard or operating-margin violations, and no unresolved sampled clearance; the minimum articulated lower bound was 0.14513 m. C4 selected 3/174 outbound holds and 22/287 return holds. Its 25 holds comprise nine score losses, two coverage overrides and fourteen planned arrival-settling holds. Median/p95 planning latency was 0.7665/0.7818 s.

This first rejected outbound arrival exposed a reader accounting defect: leg boundaries were derived from *passed* arrivals, causing the original report to allocate both legs to outbound and report a negative return duration. A read-only erratum uses logged phase transitions for segmentation while preserving physical success flags and every original report. Corrected C4 outbound path/time/SPL are 7.12455 m / 71.5 s / 0; return values are 7.94056 m / 116.9 s / 0.72934. Return-leg success is reported independently and does not imply successful beacon retrieval. All four earlier pilots' leg quantities are unchanged exactly. Each pilot has an `episode_evaluation_leg_accounting_v2.json` addendum; the prospective reader is corrected. No physics, model calls or clearance computations were repeated.

All five serial pilots are now complete. The fixed two/four-owner C3 replay checks are running; no development cohort, oracle gate or validation episode has launched.

## Two-owner throughput check

Both concurrent C3 replays passed: each matched 4,776 RGB images, 588 frozen prediction calls and selected actions, all 11,936 final command intervals, and every native physics value exactly. Together they replayed 477.44 simulated seconds in 2,867.05 wall seconds: 6.005 wall seconds per aggregate simulated second, about 1.35 times the serial-source throughput. The source measurement includes recording and the replay measurement includes verification, so this is an operational planning comparison rather than an isolated concurrency-effect estimate. The pre-declared four-owner check is running. Its completion is required before the prepared budget report admits a concurrency speedup.

Preliminary recording-size extrapolation identifies a storage blocker: roughly 85 GiB for the reduced design's remaining science recordings at full 480-s mission duration, versus roughly 3 GiB above the required RecoveryStorage reserve. This is a stated full-budget workload scenario, not a population-average prediction. The active replay checks retain small reports and remain within reserves. No cohort is queued after them.

A read-only C4 arrival diagnosis identifies a goal-reference mismatch: the episode instruction is expressed relative to the nominal starting pose, while the controller's initial frame is established after settling. During the rejected dwell, the fixed beacon expressed in that settled frame differs from the instructed target by 29.768–29.846 mm. The initial-frame physical radius check passes (maximum 19.845 mm), but the fixed-world check fails (43.894 mm). This is a target-registration mechanism in the shared testbed, not an isolated demonstration of a supervised representation defect. The fixed-world failure remains a failure. Evidence is `runs/v0_pilot_C4_dev00_ep0_attempt001/arrival_reference_diagnostic.json`; no target, controller, threshold or trajectory was changed. Any shared-harness change still follows the prescribed C1 development screen and dominant-mechanism rule.
