# Online episodic route hypotheses: implementation and actual-event replay

Implemented the memory component needed by the whole-task prototype. This is
software/interface progress, not a new navigation success or an executed return.
The previous physical panel remains 2/4 successful for every arm.

## Implemented behavior

`lewm/memory/episodic_route_hypotheses_development.py` accepts strict policy-only
RGB/body packets and the existing uninterrupted causal relative-gyro attitude.
It allocates visit-event IDs online, binds RGB bytes and body histories, and
retains initial, departure, arrival and selected scan/alignment observations.
Context grouping is temporal: a scan is not assumed to preserve physical position.
No simulator pose, cell label, map, target location or future image enters memory.

Each prior visit remains a possible appearance association alongside UNKNOWN.
The declared 4x4 RGB block-mean descriptor ranks alternatives across acquired
views; it neither merges visits nor produces calibrated probabilities. The exact
observation responsible for each minimum-distance score is retained. This is a
simple retrieval baseline, not learned place recognition or an uncertainty model.

An outward ARRIVAL_CANDIDATE advances a provisional route stack. Its opposite
departure bearing suggests a predecessor-return observation direction. A current,
RGB-hash-bound exit candidate must match within the declared 0.35-rad bearing gate
before a return attempt can begin. That gate is an integration parameter, not
qualified association, obstacle clearance or proof of a reverse edge.

Returning provisionally advances the backtracking hypothesis, never associates
the new observation with the intended place as fact. Emptying the stack yields
HOME_CANDIDATE, with home_verified and mission_complete still false. Physical
evaluation must adjudicate return. Failed attempts and between-traversal faults
remain recorded; faults suspend routing even when a valid terminal image cannot
be obtained. No image or arrival is fabricated on that abort path.

The existing stronger ObservedExploration/DirectedTraversalGraph contracts and all
completed experiment source remain unchanged. This separate prototype does not
weaken their qualification requirements. These new implementation files have not
yet been bound into a launched physical study and remain development source.

## Verification and actual evidence

Thirty new synthetic tests passed (session 87260, exit 0, 1.67 s). Coverage includes
outward and return sequences, ambiguous identical views, multiview retrieval,
fresh/hash-bound proposals, episode/reference changes, rewritten data, stale/future
observations, invalid rotations, privileged fields, copied snapshots, lifecycle
errors and failure/abort preservation. The full explicit suite passed **946 tests
across 82 files**, session 17407, exit 0, 35.63 s. These are software checks, not
maze or hardware qualification.

Read-only checker:

`scripts/check_go2_episodic_route_memory_development_v1.py`

Its final replay passed in session 92960, exit 0. It verifies the exact prior
launch/result/full-audit identities and all 215 source, 176 input and two gait
bindings, plus the selected trial's RGB/history/event files. It uses the already
audited causal attitude and executor events; it does not independently reintegrate
every gyro sample or re-audit all physical decisions. It neither runs physics nor
rescores any outcome. Earlier read-only development checks 33000 and 37732 also
finished exit 0 before multiview retention was added; their summaries do not
describe the final source below.

Post-document verification in session 80595 also passed, exit 0, confirming the
same final source hashes, predecessor bindings and four event replays. No
collection, audit or check remains running.

The four fixed-forward/both-policy trajectories supply 29 actual event RGB packets,
11 visit events and seven outward attempts. One additional terminal policy packet
supplies the native-stop time, without creating a view or arrival.

| Existing fixture | Acquired views per visit | Provisional route depth | End state | Prior physical task |
|---|---|---:|---|---|
| 00 | 3 / 4 / 1 | 2 | Return observation intent available | False second arrival; failure |
| 01 | 3 / 4 / 1 | 2 | Return observation intent available | Success |
| 02 | 3 / 2 | 1 | Suspended after scan contact at 15.908 s | Failure |
| 03 | 3 / 4 / 1 | 2 | Return observation intent available | Success |

All four retain zero trusted edges, unknown place identities and no mission
completion claim. The false arrival remains in the runtime hypothesis; no
evaluation-only geometry was used to correct it retrospectively. This explicitly
preserves the risk that a later return intent starts from an incorrect hypothesis.

### New observation: identical images at different arrival locations

Fixture 01's `rgb_0074.png` (8.9 s) and `rgb_0227.png` (24.2 s) are byte-identical,
featureless images. Their PNG SHA-256 is
`563eadee3a03e79e272146c400fac80f65dfc3a3a74609ff09f886a3c1db652f`;
their decoded RGB hash is
`da1dddbaf97ba92dc93544e31e967258218c5f27656dca912fce720b0143dea8`.
The separately audited camera records place the optical origins at approximately
(1.556, 0.185, 0.358) m and (1.375, 1.680, 0.360) m with different headings.
An actual image was visually inspected. These RGB observations alone cannot
distinguish those locations, regardless of the image encoder used. Do not infer
a camera malfunction or general renderer failure from this pair alone.

This finding motivated retaining actually acquired scan/departure views rather
than relying solely on the stopped arrival frame. Multiview retention does not
yet establish successful disambiguation; it makes additional evidence available
for the next live prototype without inventing identities.

## Reproducibility identities

Final source SHA-256:

- Memory: `01bb5ce6c9d80ad91b4ac201c65a9ff717df980bb46acf3e12c8b23e2d7e8e99`.
- Tests: `02bea54d07c6656ba752b9f73dca6c32d7786d8a585af8915d443514fd0cbd71`.
- Read-only checker: `4dd8bd1820a09f4fc19a50bf0ecedd43609c35a239c64dc5061d2fd4ee8a692a`.

Final memory snapshot hashes, fixture order 00–03:

- `d50a3317d8dc4d3f1a81cee26268b9769525d73d35d15261680727c4542b8b8d`.
- `100d57e82155a13c559f094134c586578a21e2b519b9b38eedf68de797e08e6c`.
- `4e36b0763977bcdbd02887b84b1f4291c066221d29163c0e5a76af8740b85bbe`.
- `eb68502d34142d2659461e972c19674ce49c13e618acd5272c515da2cc2388ef`.

The checker prints the exact selected image indices and summaries. It does not
create a new experiment root or overwrite predecessor evidence.

## Immediate next work

1. Add a collision-preserving physical marker to fresh development scenes and
   detect its identity from current RGB. Declare a simple marker baseline; test
   actual positive, absent, occluded and distractor frames before claiming beacon
   perception. A color tag alone cannot distinguish an identically colored decoy.
2. Connect memory and actual beacon detections to a continuous whole-task
   controller, including observed branch choice, bounded traversal, fresh return
   reobservation and independent physical home evaluation. The latest local
   controller still stops after two legs; this replay did not extend its motion.
3. Fix a fresh small connected-maze population and memory ablation before launch.
   Retain the narrow-maze arrival/clearance failures, use the same executor/sensors
   across the memory comparison, and report false home/beacon claims and unfinished
   returns. Do not call that comparison a JEPA contribution.
4. Continue the original plan's observed geometry, action-coverage and matched
   predictive-training/rollout comparisons, independent novel mazes, robustness
   and bounded real-platform evidence. The full scientific goal remains active.
