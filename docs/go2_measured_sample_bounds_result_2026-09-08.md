# Measured-return and foot-sphere refinement: completed diagnosis

The recorded-data diagnostic reproduced every original candidate surface check
at 65 forecast contexts from direct-039 commitment-pose and nominal-action
recordings. All 328 available memory updates were reconstructed; 20 unavailable
updates remained unavailable. The parallel index encloses all 6,209,795 inserted
returns, retains every voxel's first witness, and expands bounds when later
samples arrive. No ground returns or mixed observations were removed.

Five analytic tests passed in 0.16 seconds. Tight bounds use outward rounding;
sphere distance retains tangency with a 1e-12-m numerical allowance. The diagnostic
does not assert continuous surface coverage or sensor/pose uncertainty bounds.

| Recording | Whole-voxel conflicting candidates | Tight-bound AABB | Exact spheres / other AABBs |
|---|---:|---:|---:|
| Commitment direct 039, 48 contexts | 126 | 101 | 99 |
| Nominal-action direct 039, 17 contexts | 8 | 7 | 6 |

At nominal-action tick 83 the original filter rejects all six candidates. Tight
bounds remove the right-turn conflict. Exact foot-sphere geometry additionally
removes the hold conflict; forward, both arcs and left turn retain conflicts.
The current measured posture's front-foot AABBs also overlap sample bounds,
while the exact foot spheres do not. This demonstrates overapproximation in
these specific whole-voxel/AABB checks, without waiving a foot contact or claiming
clearance from unmeasured terrain.

At commitment tick 238, tight bounds remove both turn vetoes and the reported
lower-calf intersections. Forward and both arcs still have foot conflicts after
sphere refinement, while hold remains without a reported conflict. A blanket
floor exemption would discard evidence that this narrower refinement preserves.

The first 40 frames per recording took 7.735 seconds with one thread and 7.294
seconds with two, with exactly equal results; two threads were selected. Total
post-launch work took 54.540 seconds. Hardware checks recorded 82.53 GB available
RAM, 87.74 GB artifact free space and idle GPUs. The result binds 972 source
files. No native execution, model training, command change or navigation
qualification occurred in this diagnostic.

Artifacts in `go2_measured_sample_bounds_v1_attempt_001`:

| Artifact | SHA-256 |
|---|---|
| `launch.json` | `0977c96f541ea73d12ac2439341bbb1377d878a5cf7b46e8b0ffcac35c7e07a3` |
| `result.json` | `9bd03cc52dfd19fa08faf9f15bab739cd2a4a681cdf491fe842b2590f09d48df` |
| `workload.json` | `19630be84f3ad441a576fece2064d33e0cfc5f4ccaf114c04044567ebaa6d2c7` |
| `commitment_direct_039.json` | `5fe48e18beb454ac005023616a0880b6469b6e4488fe461b136d9cf5b27b6f04` |
| `nominal_direct_039.json` | `4df77d9a8b824299191d63cb10443db624e0301143fe1f2d3b108dfc909a0ce8` |

The justified next native change is this all-return bound/sphere representation
with the remaining conflicts and nominal action constraints preserved. The
full novel-maze goal remains active and unfulfilled.
