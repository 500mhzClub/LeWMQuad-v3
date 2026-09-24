# Retained depth patches explain some coarse floor-grid vetoes

The diagnostic reconstructs all 244 available direct-039 map and classification
receipts and all six candidate surface checks at 48 forecast contexts exactly.
Ten terminal-drain updates remain unavailable. It retains 299,827,200 bytes of
exact int32 invalid-pixel prefix arrays with every original RGB/depth/frame and
observed-pose identity. No future image enters a coverage query.

Five analytic tests passed in 0.47 seconds, including invalid interior pixels,
unseen regions, wall returns, history gaps, retained witnesses and batch equality.
Each positive coverage result requires the entire enclosing 44-mm foot square
to project within range/image bounds and pass every original pixel validity,
normal, planarity and 10-mm height test in one retained observation. It does not
infer coverage from the plane or interpolate across missing pixels or frames.

The original 66 conflicting candidates become 28 conditional conflicts when
original measured-grid positives are retained and complete footprint-specific
patch witnesses are additionally accepted. At tick 103, left arc's right-rear
foot has a complete witness from frame 45, and left turn's left-front foot has
one from frame 36. Both formerly blocked candidates become conditionally
admissible. Forward remains blocked there. At tick 238, left turn again has a
frame-36 witness, but left arc's right-rear footprint still lacks a complete
patch and remains vetoed. These distinctions preserve actual missing coverage.

This demonstrates overapproximation in specific 5-cm-grid requirements, not
general support safety or executed improvement. Original floor-grid cells and
receipts are unchanged. Non-floor/unknown and non-foot conflicts remain intact.

The first 40 frames took 10.246 seconds with separate foot queries and 9.941
seconds with batched queries; outputs were exactly equal and batching was
selected. Post-launch work took 87.722 seconds. Preflight recorded 82.37 GB
available RAM, 85.67 GB artifact free space and idle GPUs. This single causal
history ran sequentially. The result binds 1,000 source files. No native
execution, training, contact-policy change or navigation qualification occurred.

Artifacts in `go2_retained_floor_patch_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `6732fcad8d12565e44ac878dbd648070b5edf7f77403d9a9a3a772df2853932b` |
| `workload.json` | `3700635d57573753b8eada9f72ece99b563f76d52dc840b8cd7864307b131b5d` |
| `coverage.json` | `6f73f94832c157e986b8d4c599b4d4a1ca501d32f28eb14bfda6058cc0e401a3` |
| `result.json` | `5c4e6c4711f1850b8f29b67c4280eb4947c13f0f83f3ee382962986eab926c31` |

The justified prospective change is this additional footprint-specific coverage
witness on the affected direct-039 case. The full navigation goal remains active.
