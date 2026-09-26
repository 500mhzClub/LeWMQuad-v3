# Navigation capability: reference and retention erratum — 26 September 2026

Authority: the user's explicit resume instruction, including the new structural-only sealed-set check, sensor-hash retention and environment pin. The original preregistration and pilot outputs remain unchanged. No new navigation cohort has launched.

## Target-reference diagnosis

The generator assigns a nominal world home pose and fixed world beacon. The scene initializes the robot at that home, then executes 1.5 seconds of settling. The mission instruction remains expressed relative to the nominal start. The controller establishes its visual coordinate origin after settling. The legacy physical arrival reader uses that same settled initial-body frame; the capability reader additionally checked the registered world targets.

All five pilot missions have the same initial displacement: **2.908 mm in world x, −14.916 mm in y** (15.197 mm translation), **0.315894° yaw**, and nonzero roll/pitch. During C4's beacon dwell, the nominal mission instruction differs from the true fixed beacon in the settled initial-body frame by **29.768–29.846 mm**. The generator and fixed-world physical check agree. The controller's instruction and the legacy initial-body reader refer to a displaced target region.

This is confirmed for all five pilots on episode 00/0. Every registered episode uses the same initialization code and is potentially affected; actual settling displacements for unrun episodes have not been measured.

The existing mission is a positional beacon instruction, as already declared in the preregistration. This correction adds no visual beacon detector.

## Structural check

The zero-physics checker read only the exact maze/episode files bound by the new registry. Its access to the new sealed set was explicitly authorized by the resume instruction; no historical sealed material was accessed. Sealed geometry was neither rendered nor disclosed, and only aggregate results were retained.

| Set | Episodes checked | Registered geometry/rules pass | Nominal coordinate agreement |
|---|---:|---:|---:|
| Dev-tune | 20 | 20 | 20 |
| Validation | 40 | 40 | 40 |
| Sealed test | 120 | 120 | 120 |

Checks include endpoint clearance, occlusion, inflated connectivity/geodesic minimum, recorded shortest-path distances, and beacon/home reconstruction under each nominal start transform. **No registered episode requires replacement.**

A separate initialization-contract test transports the first pilot's measured settling displacement under each registered start transform. The uncorrected instruction fails that contract in all 180 cases, at a fixed 1-nm arithmetic tolerance. This demonstrates the reference defect; it does **not** claim measured settling dynamics for unrun episodes. The post-fix test remains pending implementation of the shared visual-start anchor.

Receipt: `target_reference_check_2026-09-26/pre_fix_result.json` under the capability output root. Checker: `scripts/check_go2_capability_episode_references_2026_09_26.py`.

## Evaluator correction and pre-fix pilot reread

The corrected reader evaluates the registered fixed world beacon/home directly, retaining the original 40-mm radius, one-second dwell, 50-mm/s quiet-motion threshold and zero-command requirement. It removes the obsolete initial-body distance as a second geometry acceptance gate. The legacy reader remains available for diagnostics; no thresholds were loosened. This is an evaluator correction, not a new harness version.

| Pre-fix pilot | Beacon | Home | Round trip | Beacon maximum dwell distance | Verdict changed |
|---|---|---|---|---:|---|
| C0 | Pass | Pass | Pass | 25.265 mm | No |
| C1 | Pass | Pass | Pass | 30.453 mm | No |
| C2 | Pass | Pass | Pass | 16.572 mm | No |
| C3 | Pass | Pass | Pass | 33.813 mm | No |
| C4 | Fail | Pass | Fail | 43.894 mm | No |

Each original report remains intact. New `episode_evaluation_reference_v3.json` files explicitly label these as **pre-fix development pilots**. Safety evidence is reused unchanged. No new physics or model computation was needed for these rereads. They do not establish corrected-harness capability or count as a corrected-harness gate.

## Shared initialization correction remains unfinished

The brief and preregistration prohibit true current pose in the controller. Simply converting both mission cues with the simulator's settled pose would add a one-time privileged initialization input. The agent has asked whether to retain the no-true-pose rule and anchor visual tracking before settling using existing cameras, or explicitly allow that one-time mission-target transform without online true-pose feedback.

No privileged transform has been implemented. No navigation-outcome-driven harness change, controller-specific rule, model change or pose reset has been made. The default is now visual anchoring before settling, preserving the existing no-true-pose rule; no new permission is needed for that choice. Its implementation, deployed correctness freeze and post-fix consistency pass remain unfinished. The geometry test alone is not permission to bypass those requirements.

## Sensor regeneration and environment

The replay owner reconstructs the existing scene from its seed, repeats the recorded settling and command tape, and uses no saved initial or per-decision snapshot. It compares native traces exactly, both cameras' native RGB/depth hashes, consumed noisy-depth packet hashes, and retained RGB PNG bytes. All existing pilot artifacts remain intact. These are implementation checks, not new navigation trials.

C0 lacks original source image/hash evidence. Even an exact C0 physics replay cannot establish bitwise image fidelity, so its future retention falls back to full RGB-D plus consumed-frame hashes. Native depth and the existing exact noise recipe permit reconstruction of the consumed noisy packet. C1–C4 may use hashes only after their complete replay passes. No per-decision snapshot persistence is needed.

The new retention mixins record hashes at acquisition and release only recording references; controller-owned live arrays are unchanged. Two focused tests check that invariant and detect subsequent RGB/depth pixel changes. The mixins are prepared but not yet wired into a new frozen navigation owner.

`docs/go2_navigation_capability_environment_pin_2026-09-26.json` records the interpreter, installed package versions, Genesis/PyTorch/ROCm, graphics driver packages, loaded AMD driver identity, devices and frozen code bindings. No environment change is authorized silently. An unavoidable change must be recorded and followed by one complete pilot replay verification before continued navigation.

The first two C1 replay launch commands failed during imports before scene construction because their Python module paths were incomplete. Their logs are preserved. The third command used the complete repository module path and began the only C1 replay physics attempt. C4's final replay additionally measures the existing lossless native-depth archive format in memory for fallback storage sizing; those compressed arrays are discarded and acceptance criteria are unchanged.

Replay receipts: `sensor_regeneration_2026-09-26/` under the capability output root. The final replay/budget addendum is recorded after that queue completes.

## Completed replay and budget checkpoint

| Controller | Replayed camera pairs | RGB matches | Consumed depth matches | Retention status |
|---|---:|---:|---:|---|
| C0 | 1149 | 0 | 0 | Full frames: original sensor evidence missing |
| C1 | 1113 | 2226 | 2226 | Hashes qualified |
| C2 | 1253 | 2506 | 2506 | Hashes qualified |
| C3 | 2388 | 4776 | 4776 | Hashes qualified |
| C4 | 1885 | 3770 | 3770 | Hashes qualified |

Every replay reproduced the complete native trace exactly. C1–C4 collectively
matched **13,278 RGB frames** and
**13,278 consumed depth packets**. C0
is missing evidence, not a demonstrated rendering divergence. The new sensor-hash
retention implementation has not yet been enabled in a navigation owner.

The measured C4 native-depth archives average 313,063 bytes per camera frame;
the maximum is 494,151 bytes. Sizing C0's two-camera 10-Hz fallback from that
maximum, retaining all logs, adding 15% storage contingency and a 4-GiB video
allowance yields **207.07 GiB**
additional storage versus **101.32 GiB** usable.
The shortfall is **105.76 GiB**.
This is a measured projection, not an assertion that all future frames attain
the maximum. It is still not a mathematical upper bound on unseen views.

Projected wall time is **118.26 hours** against
120 hours, including elapsed conservative calendar accounting, full 480-s
future missions, a fresh corrected C1 screen and C0 gate, the already-reduced
20-episode validation set, prior 15% time contingency and one extra hour for
replay/capture costs. Additional harness iterations or substantial initialization
overhead would require another projection.

**Stop before the cohort:** the storage projection does not fit. The reader fix,
pilot rereads, structural diagnosis and replay qualification are complete. The
shared visual-start correction, post-fix test and new owner freeze remain
unfinished. Nothing has entered validation or the oracle gate. No additional
artifact retirement, physics retry or navigation cohort was launched.

A possible storage resolution is to retain the first already-scheduled corrected
C0 gate episode in full, verify its complete sensor replay once, and switch only
subsequent C0 runs if that check passes. This would add no science episode, but
it is a conditional retention proposal, **not an executed or qualified change**.
The C1 gate prerequisite and all safety/capability thresholds would remain.
If C0 cannot be qualified, keep its full recordings and stop for storage before
proceeding. Otherwise more capacity is required; no further deletion is assumed.

Numerical receipts: `go2_navigation_capability_sensor_regeneration_2026-09-26.json`
and `go2_navigation_capability_resume_budget_2026-09-26.json` in `docs/`.
