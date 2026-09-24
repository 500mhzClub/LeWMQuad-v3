# Go2 G3 exact-physical equivalence V1 result

Date: 2026-07-13

Status: **FAIL; immutable V1 output, no gate relaxation**

## Bound output

- artifact:
  `.generated/go2_g3_exact_physical_equivalence/v1/candidate.json`;
- file SHA-256:
  `b7176cca80306768c6c851c61c2ba31636093b15bae777b1966cb2d56edc3d4c`;
- canonical content SHA-256:
  `070392510e976ca753414ab3881d8240654d152d9e8197b1f689c8c39c26f4aa`;
- reviewed source graph SHA-256:
  `0ec6f7194fae94eecaecdf9a4d2500164275a3023d08e8115e31273d8ae43009`.

The reviewed isolated launcher evaluated the fixed 24-scene development
validation set with six CPU workers. It opened no RGB, checkpoint, G2,
held-out, sealed, runtime, or GPU input.

## Result

| Requirement | Result |
|---|---:|
| Development scenes | 24/24 |
| Beacons represented | 96 |
| Independent discrete label equality | 24/24 scenes; 0 mismatched cells |
| Independent component equality | 24/24 scenes; 0 mismatched cells |
| Deterministic A* distance probes | 192/192 |
| Unsafe configuration-FREE cells | 0 |
| Scenes retaining every claim endpoint | 22/24 |
| Claim endpoints retained | 90/96 |
| Candidate conservative-equivalence gate | **FAIL** |
| Legacy strict-binary equality | **FAIL**, 0/24 scenes |

The six lost endpoints are confined to two scenes:

- `go2_deployment_medium_maze_3e28c26ef602`: `2/4`; the registered morphology
  reduces the spawn component from 4,023 analytic-safe cells to 1,283 cells;
- `go2_deployment_medium_maze_5689fb82c098`: `0/4`; the registered morphology
  reduces the spawn component from 3,632 analytic-safe cells to 143 cells.

The analytic 0.47 m disc-clearance component retains `4/4` endpoints in both
scenes. The failure is therefore not endpoint centre sampling, line of sight,
planner disagreement, or an unsafe-free defect. It is conservative topology
loss caused by the registered 0.10 m full-square physical raster followed by
the 89/69 configuration morphology.

## Pre-output-independent correction basis

The selected successor is the preregistered
[`0.05 m` physical / `0.10 m` configuration V2 design](lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md).
It preserves the V1 planning lattice while deriving 316 FREE-support and 276
OCCUPIED-support physical offsets under exact shared-origin cross-grid
alignment.

After the immutable V1 result was recorded, a read-only, non-authoritative
six-process CPU diagnostic applied that design to the same 24 development
scenes. It retained `96/96` claim endpoints across `24/24` scenes, admitted
zero unsafe FREE cells, and reduced conservative false rejects from 33,338 to
13,260 on the unchanged `0.10 m` configuration lattice. The two V1 failures
each retained `4/4` endpoints. Per-beacon, topology, support-hash, and
derivation evidence are frozen in the V2 design contract.

A same-grid `0.05 m` diagnostic was also safe and retained `96/96`; it is not
selected because it needlessly doubles planning resolution on both axes and
nominally quadruples A* state density. Its 313/277 kernels are not the selected
316/276 cross-grid kernels.

These figures are diagnostic design evidence only. They do not amend or
replace V1 and do not authorize learned G3 output.

## Required successor

G3 V2 must implement the frozen two-resolution contract, update memory and
exact-adapter bindings without accepting caller-created evidence, and receive
different-agent source review. It must write
`.generated/go2_g3_exact_physical_equivalence/v2/candidate.json` and repeat all
V1 safety, equivalence, A*, endpoint, source-capture, and result-binding checks.
The V1 output is never deleted or overwritten.
