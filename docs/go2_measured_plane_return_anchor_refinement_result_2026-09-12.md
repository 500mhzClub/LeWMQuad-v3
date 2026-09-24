# Recorded return-anchor refinement diagnostic

The fixed diagnostic completed on the three previously qualifying chained
image pairs. Two pass the existing floor-plane refinement and global motion
envelopes; one is rejected because the refinement loses an original image
inlier. This is component evidence conditioned on the audited baseline visual
state, not a recovered observer or a completed navigation run.

| Camera | Reference → current frame | Outcome | Retained inliers | Refined RMS |
| --- | --- | --- | --- | --- |
| Primary | 3100 → 3103 | Pass | 19 | 0.337548 mm |
| Primary | 3099 → 3103 | Reject: original image inlier lost | No refined fit admitted | — |
| Auxiliary | 3100 → 3103 | Pass | 37 | 0.681497 mm |

Both passing fits retain the complete original accepted inlier populations
and gate values. Refinement changes their original image-only global positions
by 0.105013 mm and 0.061217 mm respectively. No bridge allowance, threshold,
retained reference, or live controller was changed.

## Evidence and scope

The diagnostic authenticates the completed native result and the earlier
64-pair diagnostic. It checks the same 95 raw input hashes against both the
original pair launch and the completed native artifact roster, before and
after execution. It verifies all 2,566 bound sources before and after execution.

It reconstructs the original 22 paired feature frames and public relative
gyro interval, reproduces every feature witness and each of the three original
endpoint fits exactly, and retains the original inlier masks. It reconstructs
the floor planes at frames 3099, 3100 and 3103 from the paired recorded depths,
requiring exact equality with the recorded measured-plane receipts. The prior
visual attitude supplies the same floor extraction direction as the original
observer. The selected baseline visual poses come from frames 3099, 3100,
3102 and 3103 in the authenticated 3,124-observation decision stream.

The probe applies the original global motion envelopes and existing
`measured_plane_dual_camera_pose_development.refine` function. It preserves
all three outcomes. It does not compare the candidate anchor with the current
increment or other anchors, replay the complete observer, promote a reference,
admit a pose, load a model, or execute physics. The original diagnostic's
negative results at frame 3113 remain negative; this probe does not retest or
replace them. No navigation recovery follows from these component results.

## Execution identities

Artifact root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_measured_plane_return_anchor_refinement_v1_attempt_001`.

| File | SHA-256 |
| --- | --- |
| `launch.json` | `a91d38178661acc2ab9ac143e8434c6455246b8d89f77ec85abb3730da8aeffd` |
| `result.json` | `403c5d3ffc8e8d17821b1891d92f07ad36e2ba90773b742919eb28d7117b9403` |
| `scripts/probe_go2_measured_plane_return_anchor_refinement_v1.py` | `f679d7182318a1ee9176ddfb3ee870d16c683379d4242818c4f63b5c5703cac4` |
| `lewm/tests/test_measured_plane_return_anchor_refinement_development.py` | `5c88513c3f8e2a4336d1dd195ec4069a0643742d1a9299252c48af67f9aeb9e1` |

The completed native input result is
`4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18`;
the original pair result is
`d3a01cc5b667b7038964adf04ee135179af8a35c7d8bf0ed875b5baae5831c6e`.
The diagnostic exited successfully in 8.0366 seconds, with one OpenCV thread
and no automatic retry. Preflight passed before the exclusive output was
created. All 18 focused tests passed, including actual static RGB-D fits in
both camera coordinate systems, inconsistent valid floor geometry, original
evidence tampering, population changes, and the global displacement gate.
During test development, a deliberately changed plane offset first failed
receipt integrity instead of testing geometric rejection; the fixture was
corrected to construct a complete valid conflicting plane before execution.

## Next action

The full-history timing replay remains live and owns the single full CPU
replay slot. At the completion check for this diagnostic, frame 2204 matched
both complete decision comparisons and unchanged public inputs. The existing
chained-controller waiter remains live; its child has not started. That replay
must establish the actual first changed controller request or terminal
decision before any separate candidate physical run can be prepared and
admitted. This diagnostic does not bypass that dependency.
