# Direct corner-flow association on fixed failed pairs

A separate association method passed the unchanged rigid-fit checks in three
of four camera pairs that the original matcher rejected. These are selected
development failures, not a general tracking benchmark or admitted controller
poses. Temporal continuity and fresh navigation remain untested.

The method tracks the existing reference corners directly with pyramidal LK,
starting from the same-pixel hypothesis. It retains the original0.5pixel
forward/backward flow threshold and depth lifting. It replaces independent
current-corner/mutual-descriptor/location association gates with an explicit
11×11patch normalized-correlation threshold0.90 and minimum patch standard
deviation2intensity levels. Duplicate reference/current locations are removed.
Reference population is capped at600. These are declared association changes;
the original rigid geometry thresholds are unchanged.

Nine synthetic tests passed in0.21s. They include known translated texture
passing the original rigid fitter, empty/invalid-depth support, duplicate
corners, flat/inverted/border patches, bounded inputs, nonfinite flow and reverse
inconsistency. They do not establish broad false-match rates.

Actual diagnostic session69292 completed normally, exit0. Root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_direct_corner_flow_failed_pairs_v1_attempt_001`.
Result SHA-256:
`6525b5e02975def2a296b64ca630bcd55a9da63592f4fbe8f0c828e8820070db`.
Launch SHA-256:
`cb1f721e935622a69794fd444bf5e8ae9fe0b8c6d95ce511885dff3d8ede7f5a`.
1661source and3071input bindings were checked before/after. Work5.640770641854033s.
Both cases use their final pre-failure reference and first failed current frame.
The original four rejection reasons were reproduced exactly.

| Case/pair | Camera | Original lifted matches | Direct-flow lifted matches | Original rigid gates on alternative | Inliers / fraction | Post-hoc translation error |
|---|---|---:|---:|---|---|---:|
| Maze1,213→214 | Primary | 10 | 22 | Reject: fraction/grid/displacement gate | — | — |
| Maze1,213→214 | Auxiliary | 12 | 42 | Pass | 31 /0.7381 | 0.1342mm |
| Maze3,263→264 | Primary | 6 | 13 | Pass | 12 /0.9231 | 0.0644mm |
| Maze3,263→264 | Auxiliary | 11 | 27 | Pass | 20 /0.7407 | 0.5927mm |

Reference/current inlier-grid coverage for successful fits was9/9,9/8and7/8;
residualRMS0.3702mm,0.2290mm,0.4477mm. Post-hoc rotation errors were
0.0006705rad,0.0000404rad,0.0008427rad. All three pass the existing body
increment translation/rotation envelopes. Their fitted rotations also pass
the original public-gyro consistency gate. Primary maze1 remains rejected;
the combined fraction/grid/displacement rejection was not further separated
in this diagnostic.

The gyro comparison uses the original validated public fast-gyro packets across
the100ms pair. Auxiliary fitting uses the existing fixed reference-frame
adapter, then converts the fit back to body coordinates. Native poses are
opened only after association and fitting, for post-hoc error measurement;
neither algorithm receives native position/orientation. No camera, model,
command, physical episode or original outcome is altered.

Source SHA-256:

- `lewm/direct_corner_flow_association_development.py`:
  `3007e0178d67f4af54deb1ffe2935415936354a6eebaa77e60bbbc9344e8521f`.
- `lewm/tests/test_direct_corner_flow_association_development.py`:
  `95d2022f69c7ddd124f34fdd55fdc9ee975b9a7035b8dc5c373f8d64b8a2fcaa`.
- `scripts/diagnose_go2_direct_corner_flow_failed_pairs_v1.py`:
  `2f3e6a6a6c1d1731aa59710aa34159f44b184e2425eedb01795b6d67bee56409`.

Hardware admission80.225GBavailableRAM,90.218GBartifactfree,
21.360GBworkspacefree,16physical/32logical/all32affinity,CPU3.3%,bothGPUs0%.
One brief CPU diagnostic ran beside independent-reactive worker2476369.
No new simulator scene, checkpoint or model training was used.

Next: integrate this frozen association as a separate, explicitly recorded
short-interval fallback in the full observer, preserving existing successful
decisions, measured-conflict vetoes, rigid/gyro envelopes, retained-reference
rules, temporal-continuity checks and bridge budgets. A pairwise fit must not
become a pose merely because the original observer failed. Verify full causal
replay through first intervention and then fresh physical navigation. More
matches or small error on these selected pairs alone do not prove reliability.
