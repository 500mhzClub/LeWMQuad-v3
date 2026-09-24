# Adjacent flow does not resolve the terminal correspondence failure

The diagnostic reconstructed all 230 original observer evidence rows exactly,
including terminal failure at frame 229, and checked all 229 adjacent pairs.
Counter-instrumented descriptor matching returned exactly the original point
and pixel arrays on every pair. Native pose was opened only after all sensor
pair fits had been persisted.

Both methods qualified 227 pairs. Descriptor matching failed at frames 228 and
229; adjacent gyro-seeded flow failed at 227 and 229. Flow therefore gains one
pair and loses another, and does not resolve the terminal pair. No candidate
observer is adopted or defined from this result. The earlier whole-trajectory
flow failures remain rejected and unchanged.

At frame 229, the reference/current corner populations were 50/76. Descriptor
ratio matching produced 22 forward and 25 backward survivors, 21 unique mutual
pairs, then ten bidirectional seed-consistent depth pairs. Ten is below the
unchanged minimum rigid-match count. Flow retained 48 projected seeds, 46 forward
tracks, 37 bidirectional unique tracks and 37 depth pairs, but failed the rigid
consensus fraction/grid/displacement gate. Its composite failure does not identify
which individual clause failed. More tracks did not establish a usable fit.

Across their respective qualified populations, maximum descriptor translation
and rotation errors were 2.495 mm and 0.00312479 rad; flow maxima were 1.074 mm
and 0.00120713 rad. These are conditional adjacent-pair errors using preceding
original observer references, not errors of a new complete observer or calibrated
uncertainty. They do not outweigh the unresolved terminal failure.

The result binds 1,031 sources. Preflight recorded 81.99 GB available RAM,
84.78 GB artifact storage free, 0.3% CPU utilization and idle GPUs. One causal
observer and the fixed pair comparisons ran sequentially with one OpenCV thread.
Post-launch work took 36.976 seconds. No physics, commands, training or old-result
changes occurred.

Artifacts in `go2_adjacent_pair_correspondence_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `20f8bee300ba4b52a0228c9e5e3caba5730a994a5a35d204edd979de07b9b56a` |
| `sensor_pairs.json` | `b64d50c9f90443c83b3686d7023e04fedeef058899412f16e4ff30b8c416e658` |
| `evaluation.json` | `a2b03034c2f0bd462c0c2f12025dd492746a02a3eb3300b221af2b1ebab4a572` |
| `result.json` | `7da2a7224f4805cbfa67f76de17197d3b8c543f1bfd2e7bb544928ca1924db42` |

The next bounded learning diagnosis should compare all six fixed models on the
same actually executed half-second commitments, keeping action-switch and
repeat contexts explicit. Current direct-model predictions understate turning
and sometimes predict backwards displacement while the robot still drifts
forwards after a switch. Authenticate and quantify these errors before defining
new transition collection or training. Unexecuted actions remain unlabeled.
The full goal remains active; near-goal approach is not verified navigation.
