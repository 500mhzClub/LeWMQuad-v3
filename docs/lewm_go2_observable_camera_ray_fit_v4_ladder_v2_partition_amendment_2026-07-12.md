# Observable camera-ray fit V4 ladder V2 partition amendment

Date: 2026-07-12

Status: frozen before any V4 RGB decode, model output, or checkpoint

## Failure

The new independent checkpoint verifier required exact target-partition
constants before it could authorize inference. Deriving those constants from
the already-audited train-evidence arrays exposed an impossible old ladder:

- N=1 has `9,408` no-hit rays and **zero** represented hit/depth targets. Its
  depth median and p95 gates cannot be computed.
- N=4 has no large-enclosed-maze frame because four frames cannot cover the
  five-family round-robin. Its mandatory every-family gate cannot be computed.

The earlier synthetic tests invented positive depth and every-family counts,
which hid both contradictions. No learned result was inspected; no V4 RGB or
checkpoint exists.

## Correction

The ladder is `(5, 16, 32, 320)`. N=5 is the smallest unchanged,
label-independent family-round-robin prefix containing one frame from every
registered family. It contains both pixel states and all three raster classes:

- pixel no-hit/hit: `26,323 / 20,717`;
- raster UNKNOWN/FREE/OCCUPIED: `16,123 / 4,259 / 98`;
- one frame from each of the five families.

N=5 uses `1,000` optimizer steps, batch size one, and the former N=4 numeric
thresholds. N=16, N=32, and N=320 retain their prior subsets, budgets, and
thresholds. Both registered seeds remain required, sequentially, only after
seed one passes the complete ladder.

The canonical target-partition freeze is
`docs/lewm_go2_observable_camera_ray_fit_v4_target_partitions_2026-07-12.json`,
file SHA-256
`4ca8ef7f427f525e591a107496ef3b42c2586a9e47f7b8a7a0fd5710ca0d248a`,
canonical content SHA-256
`8dd54d178e3c00a8622d89e4e371a115e1391f34588f667c20cd95b970fc68d2`.
It binds each rung's family counts, subset hash, first/last keys, exact binary
target counts, distance/family partitions, raster counts, and ordered target
byte commitments. The source implementation must reproduce this artifact from
all 180 verified dataset files before any metric verification.

The independent reproducer is
`scripts/verify_go2_observable_camera_ray_fit_v4_target_partitions.py`, file
SHA-256
`4624dd761901808c72b37eb256b360e3db61c9b8f61337879547ed38836a3eed`.
With every native numerical thread capped to one, it reverified the complete
180-file dataset inventory and reproduced the frozen artifact exactly in 4.7
seconds. It opened no RGB, checkpoint, G2, held-out, runtime, or sealed input.

All trainer, metric-verifier, checkpoint, G2, runtime, held-out, promotion, and
sealed authorizations remain false until this amendment is implemented and
independently reviewed.
