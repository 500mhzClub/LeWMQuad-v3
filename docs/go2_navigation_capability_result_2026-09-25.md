# Navigation capability: pilot results and storage stop — 25 September 2026

**The pilot is complete; the testbed and controller capabilities are not yet qualified.** Execution stopped before the remaining development cohort because its recording allowance exceeds available storage above the required reserve. No reserve was breached.

| Controller | Development pilot round trip | Disallowed contacts | Validation capability |
|---|---:|---:|---|
| C0 Oracle | 1/1 | 0 | Not evaluated |
| C1 Command history | 1/1 | 0 | Not evaluated |
| C2 Reactive | 1/1 | 0 | Not evaluated |
| C3 JEPA, maze-data head | 1/1 | 0 | Not evaluated |
| C4 Direct supervised predictor | 0/1 | 0 | Not evaluated |

These are paired development pilots on **one maze, episode 00/0**, not validation estimates or paper results. The 80% capability criterion cannot be applied to them. No validation or sealed-test episode has run or been rendered.

## Pilot outcomes

| Controller | Beacon verified at (s) | Home verified at (s) | Outbound / return SPL | Hold decisions | Episode wall (s) | Wall / simulated time | Planning latency median / p95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 64.8 | 114.8 | 0.836 / 0.914 | 3/278 | 676.10 | 5.888 | 1.677 / 1.735 |
| C1 | 68.5 | 111.2 | 0.821 / 0.928 | 4/270 | 905.59 | 8.142 | 2.607 / 2.668 |
| C2 | 85.3 | 125.2 | 0.763 / 0.946 | 0/304 | 1009.90 | 8.065 | 2.589 / 2.648 |
| C3 | 150.1 | 238.7 | 0.580 / 0.859 | 150/588 | 1937.24 | 8.115 | 2.608 / 2.667 |
| C4 | Failed | 188.4 | 0.000 / 0.729 | 25/461 | 670.57 | 3.559 | 0.766 / 0.782 |

All five pilots had zero native disallowed contacts, zero hard-clearance (5-mm) or operating-margin (20-mm) violations, and zero unresolved sampled-clearance states. All 27 collision primitives were evaluated at every native 2-ms step; the secondary FK interval checks also had no threshold failures. Minimum articulated separation lower bounds were 172.73, 155.85, 116.76, 126.15 and 145.13 mm for C0–C4, respectively.

C1/C2 source timings above include the deployed unused neural workload. Its subsequently qualified omission improves prospective runtime without changing decisions. These timing columns are therefore not measurements of an already optimised, latency-matched controller comparison. Physics remains paused during planning.

## Failure and hold accounting

| Controller | Movement outscored | Motion-dependent footprint/view override | Planned arrival-settling hold | Verified mission failure |
|---|---:|---:|---:|---|
| C0 | 0 | 2 | 1 | None |
| C1 | 0 | 3 | 1 | None |
| C2 | 0 | 0 | 0 | None |
| C3 | 100 | 2 | 48 | None |
| C4 | 9 | 2 | 14 | Beacon arrival/reference mismatch |

These are descriptive counts from different trajectories. Intended settling holds are included in the pre-registered hold fraction; they are not automatically navigation failures. Footprint exclusions depend on the motion forecast and observed floor coverage, so they are not observation-only exclusions.

**C4 did not satisfy the fixed-world beacon criterion.** It declared outbound arrival at 71.5 s. Its initial-frame physical check passed, but its one-second dwell reached 43.894 mm from the registered world beacon, outside the fixed 40-mm radius. Home arrival passed independently at 188.4 s; round-trip success remains false.

The retained data identify a shared target-registration issue: the goal instruction uses the nominal starting pose, while the initial frame is established after settling. During the rejected dwell, the fixed beacon expressed in that frame differs from the instructed target by 29.768–29.846 mm. This does not isolate a supervised representation defect. The diagnostic is `runs/v0_pilot_C4_dev00_ep0_attempt001/arrival_reference_diagnostic.json`. No target, criterion, model or controller was changed. Any harness change still follows the full prescribed C1 development screen and dominant-mechanism rule.

The first failed outbound arrival also exposed a reporting bug: using only *passed* arrivals as leg boundaries produced a negative return duration. Preserved V2 addenda use logged phase boundaries independently of physical success. C4 now has outbound 7.12455 m / 71.5 s / SPL 0, and return 7.94056 m / 116.9 s / SPL 0.72934. All success and safety flags are unchanged; the first four pilots’ leg metrics are identical to their original reports. The original erroneous report is retained. No physics or model computation was repeated for this correction. Return-leg success denotes verified home arrival, independently of beacon retrieval.

## Restoration, throughput and video

C0’s oracle check passed 1,668 candidate-prefix comparisons with zero position/yaw error. These include common pre-dispatch prefixes. The stronger executed-command coverage check covers all 5,460 admitted 20-ms intervals from 276 admitted plans inside their selected branch’s verified prefix. This qualifies the one pilot, not the 19/20 oracle gate.

C1’s unused-workload omission matched 2,226 RGB images, 270 selections, 5,561 command intervals and every native physics value exactly. C2 matched 2,506 images, 304 selections and 6,261 command intervals, also with exact native values. C2’s complete checked replay took 258.08 wall seconds for 125.22 simulated seconds (2.061 wall/sim). For scheduling, the combined C1 replay-and-video measurement conservatively bounds its optimised execution at 3.002 wall/sim; it is not an isolated source-run timing.

| Concurrent C3 owners | Aggregate simulated seconds | Group wall seconds | Wall / aggregate simulated time | Throughput relative to serial source | Exact agreement |
|---|---:|---:|---:|---:|---|
| 2 | 477.44 | 2867.05 | 6.005 | 1.351× | Passed |
| 4 | 954.88 | 5552.06 | 5.814 | 1.396× | Passed |

Every one of the six C3 checks matched 4,776 source RGB images, 588 frozen prediction calls and selections, 11,936 final command intervals, and every native physics value. Two owners improve operational throughput by about 35%; four by about 40%. Four therefore add only about 3.3% over two. Source timing includes recording; checked replay includes verification, so these are planning measurements rather than isolated concurrency effects. Concurrency is qualified for C3 on the recorded devices only; no C0/C1/C2/C4 concurrency speedup is assumed. These replays are implementation checks, not additional science episodes.

The [pipeline-test video](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/videos/pipeline_test_chase_revision002/pipeline_test.mp4) contains 3,337 frames at 1920×1080, 30 fps, H.264/yuv420p. It uses source egocentric RGB at its native cadence, a separately rendered chase view, minimap and HUD. Decision replay passed and chase pose errors were zero. The [contact sheet](/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001/videos/pipeline_test_chase_revision002/pipeline_test_contact_sheet.png) passed visual review; metadata binds the episode, harness, models and replay evidence. It is labelled a pipeline test, not an official capability example.

## Frozen preparation and model identities

- Pre-registration: commit `bb9ab232`, SHA-256 `b6a84db2f304282d9d7ec9327ac21420d6e8c3e262de20178c7f243791a887d6`, before generation or physics.
- Registered sets: 10 development mazes × 2 episodes, 20 validation mazes × 2 episodes, and 60 sealed test mazes. Same generator family as audit layouts 00–07; duplicate registry and fixed seed rules exclude those audit layouts. Registration commit `523005c1`.
- Harness: v0 controller algorithms, sensors and candidate bank remain unchanged. Adapter revision 4 is bound by `872c1e11`; reader corrections do not change navigation behaviour.
- C3: pre-declared maze-data head, chosen from existing V4.2 point estimates before any new run. No head switch or fitting followed the pilots.
- C4: 17,397,283 parameters; 8,414 prescribed training contexts; fixed seed and 1,760 updates; final checkpoint `d491584b17ee4321d65daf883460ccf3b626fe8d735b986473d2ecc477064013`. Conservative GPU-owner accounting was 4,869.74 s (1.353 h), within 12 h. No validation selection or new navigation data entered fitting.

Training-render provenance remains unverified for the current predictor/readouts and the C4 training inputs. Controller outcomes cannot by themselves support attribution to a representation defect. This pilot exercises navigation to a supplied positional beacon instruction and return; it does not qualify visual beacon detection.

The first oracle startup failure and C4 metadata-writer startup failure remain preserved. The successful C0 pilot omitted persisted camera records; its native physics and decision evidence survive, but it cannot supply a source-image-verified video. It was not repeated to repair that omission. A lossless RGB archive check saved only about 7.6% and was not adopted. No existing artifact was retired.

## Measured budget and stop decision

Planning assumes every remaining mission uses the full 480 seconds, includes native articulated analysis and video work, and adds 15% timing contingency. It covers the remaining v0 development screen, one oracle gate and qualification; additional harness versions require updated remaining-budget accounting. These are workload assumptions, not population confidence bounds.

| Design | Remaining science episodes | Projected total wall hours | Additional recording/video storage |
|---|---:|---:|---:|
| 2 validation episode(s) per maze | 198 | 149.35 | 142.32 GiB |
| 1 validation episode(s) per maze | 118 | 96.20 | 86.11 GiB |

The programme had consumed 4.85 hours of its conservative elapsed window when measured. The full design exceeds 120 hours, so the brief’s reduction rule selects **one validation episode per maze**, retaining all 20 mazes: IDs 10/0 through 29/0. C0 uses the lowest ten IDs in that set, 10/0 through 19/0. These identities were fixed before any validation outcomes. The 80%/zero-contact capability criterion is unchanged (at least 16/20 round trips for C1–C4).

**Storage blocks the next cohort.** RecoveryStorage has 15.18 GiB free. After the 12-GiB reserve and existing 128-MiB closeout guard, 3.05 GiB is usable. The nine remaining C1 development episodes require a 6.86-GiB full-budget recording allowance: a 3.81-GiB shortfall. A single short episode might fit, but the planned screen cannot be admitted under that allowance. Workspace has 5.49 GiB free against its 4-GiB reserve and is not a substitute large-artifact destination.

The selected remaining programme projects 86.11 GiB of recordings/videos, a 83.06-GiB shortfall. Roughly 4 GiB of additional free capacity would admit the next development screen; roughly 90 GiB would cover the current one-gate programme projection with some room. Further harness versions would need a revised storage allowance. Even the illustrative scenario where future science episodes merely match these pilot durations needs 26.03 GiB before new videos. No deletion or alternative-volume move has been performed.

The finite blocker is storage admission, not C4’s controller failure. All pilot evidence, qualified replay paths, fixed models, episode assignments and the pipeline video remain valid within their stated limits. The automatic pilot owner has completed and queued no cohort.

## Remaining work

1. Restore sufficient storage capacity and recheck the remaining wall budget. Reuse the completed C1 00/0 screen episode, C0 00/0 oracle evidence and fixed C4 checkpoint.
2. Run C1 on the nine remaining first development episodes. Classify failures, then follow the one-change-per-version rule if needed, with at most six versions.
3. Reach C1 ≥9/10 and pass the unchanged C0 gate: ≥19/20, zero disallowed contacts and zero hard violations. One oracle pilot does not pass this gate.
4. Freeze the passing harness and run the fixed reduced validation assignments, including C0’s ten-episode subset. Produce per-maze results and 95% maze-cluster intervals.
5. Produce the official C1–C4 videos, any applicable composite, sheets and bound metadata using the pre-declared episode selection. Complete the capability report and E1 proposal, then stop. E1 execution remains unauthorised.

## Evidence

- [Authoritative brief](go2_navigation_testbed_controller_capability_agent_brief_2026-09-25.md)
- [Pre-registration](go2_navigation_capability_preregistration_v1_2026-09-25.json)
- [Measured pilot budget and bound results](go2_navigation_capability_pilot_budget_2026-09-25.json)
- [Development progress and preserved errata](go2_navigation_capability_kickoff_progress_2026-09-25.md)
- Runtime root: `/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001`.
- Per-pilot results: `runs/v0_pilot_C*_dev00_ep0_attempt*/episode_evaluation_leg_accounting_v2.json` (C0 attempt002; all others attempt001).
- Completed replay package: `concurrency_C3_v0_attempt001/result.json`.
- Stop receipt: `programme_storage_stop.json`.
