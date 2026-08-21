# H1 articulated swept-geometry sufficiency V1

## Decision

**`FULL_ARTICULATED_GEOMETRY_CONTACT_PROXY_NO_GO`**

Secondary findings are `DEPTH_FIELD_OF_VIEW_LIMITATION`,
`LIDAR_VERTICAL_COVERAGE_LIMITATION`, `SENSOR_TIMING_LIMITATION`,
`DYNAMIC_CONTACT_NOT_PURELY_GEOMETRIC`, and
`CONTINUATION_CONTACT_RISK`.

The target remains only `SIMULATED_DISALLOWED_CONTACT_PROXY` during the
committed H1 block. This is not evidence about material hazard, injury,
property damage, people, fragile infrastructure, or closed-loop safety.

## Frozen bindings and control contract

The experiment started at source commit
`ea361860afbbd814fd7110a5b8ea504ff83293b9` and reused the 24 calibration
and 24 held-out states from `WIDE_GEOMETRY_EMBODIED_CONTACT_PROXY_V1`
(12 candidates per state). The learned checkpoint
`3e556531a0442df214d0667ad42110e42806ec3aa7aa240c2b2746d7c304af31`
was identity-bound only and was never opened or executed.

Exactly one five-tick block is committed: 0.1 seconds per command tick and
0.5 seconds total. Replanning occurs after H1; blocks 2–4 are replaceable.
Hold is available on the next cycle but is not a validated emergency brake,
and H2 is not a validated stopping horizon.

## Geometry and replay contract

The bounded replay used Genesis 0.3.14 and its packaged Go2 URDF. For every
branch it captured 250 physics steps at 0.002 seconds, 13 link transforms, 12
joint positions, and 27 collision-shape transforms. The full condition used
all non-ground scene boxes and the exact URDF sphere, capsule, and box
primitives. Sphere/box distance is analytic; capsule/box distance uses 33
fixed axial samples (maximum axial discretisation interval `length/64`);
box overlap uses the 15-axis OBB separating-axis test. No contact label,
force, or impulse enters a geometric score.

Depth is the frozen ideal 64×48 front stream with 78.323-degree horizontal
FOV and 0.05–10 m range. LiDAR is the frozen ideal 360-degree stream with 180
azimuth bins and four vertical channels at −15, −5, 5, and 15 degrees. The
sensor sweeps back-project and accumulate current plus true H1 surfaces, then
query the complete 250-step articulated trajectory. Ground points are
excluded so expected support contact cannot become a disallowed-contact
score. LiDAR remains a changed deployment sensor contract.

The deterministic fixture covered clear, front, side-FOV, low-calf vertical,
between-sample, positive-separation, support-ground, self-contact, threshold
tie, abstention, and kinematic-selection cases. All passed with byte-identical
regeneration (fixture digest
`3fe93da0f979ee168c10eb625858fed07e84e44ffd111a57b066341a0f510249`).

## Materialisation receipt

- Replayed: 48 states, 576 registered branches, 144,000 H1 physics steps.
- New state/candidate identities: zero.
- Exact action traces: 576/576.
- Exact five-tick and aggregate H1 contact traces: 576/576.
- Exact H1 poses at the historical tolerance: 574/576.
- Persisted geometry: 150,021,916 bytes.
- Summed compute: 1,738.929 seconds; corrected four-worker wall run: 464.631
  seconds.

The two pose mismatches were preserved and serially reproduced:
`wide-cal-0-05:10` (46 µm x error and 0.257 mrad yaw error) and
`wide-held-1-04:09` (0.925 mm y error and 1.908 mrad yaw error). Both retain
exact actions and contact labels. Excluding both rows leaves every condition
and every gate failed; full-geometry held-out AUC/AP become 0.776568/0.359937.
The result therefore does not depend on treating either row as exact, but the
574/576 limitation remains explicit.

## Calibration and held-out result

Contact risk is `-minimum_clearance`; a clearance at or below the threshold
is rejected. Thresholds were selected on calibration only.

| Condition | Threshold m | Held-out AUC | AP | Recall/FNR | Negative retention | States retained | Selected contacts | False abstentions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full articulated scene | -0.011877 | 0.775592 | 0.358643 | 0.924528 / 0.075472 | 0.578723 | 13/24 | 1 | 11 |
| Front depth sweep | 0.846271 | 0.644721 | 0.274542 | 1.000000 / 0.000000 | 0.102128 | 2/24 | 0 | 22 |
| LiDAR sweep | 0.119241 | 0.686391 | 0.251616 | 0.943396 / 0.056604 | 0.412766 | 9/24 | 1 | 14 |
| Depth + LiDAR sweep | 0.119241 | 0.705339 | 0.279998 | 0.943396 / 0.056604 | 0.412766 | 9/24 | 1 | 14 |

Full-scene selected progress was 0.220401 m versus 0.154377 m for the
oracle-contact kinematic selector, but this value is not a safety success:
one selected branch contacted and 11 states falsely abstained. Its normalized
safe-selection regret was 0.126899 and best-negative top-3 was 0.375. Depth
selected no contact only by retaining two states. LiDAR and fused each kept
nine states, selected one contact, produced 0.193412 m mean selected progress,
0.093663 regret, and 0.291667 best-negative top-3. Later selected continuation
contacts were reported separately (two for full, LiDAR, and fused; zero for
depth); they are not H1 hard-safety violations.

## Held-out threshold frontiers

No held-out threshold for any condition satisfies its complete gate.

| Condition | Max retention at recall ≥0.95 | Max retained states | Max progress with zero selected contact | Complete gate points |
|---|---:|---:|---:|---:|
| Full articulated scene | 0.531915 | 12 | 0.230334 m | 0 |
| Front depth sweep | 0.178723 | 4 | 0.199761 m | 0 |
| LiDAR sweep | 0.365957 | 9 | 0.261419 m | 0 |
| Depth + LiDAR sweep | 0.365957 | 9 | 0.261419 m | 0 |

The failure is therefore not repaired by a different threshold on this held
panel.

## Family and contact attribution

Full-scene recall/retention by family was: large 0.428571/0.769231, loop
1.0/0.695652, medium 1.0/0.24, and small 1.0/0.509804. Retained states were
5, 4, 1, and 3 respectively; the only selected contact was in the large
family. The medium family retained effectively no route progress, while the
loop family exceeded the regret limit.

Among 53 frozen held-out H1-positive branches, the responsible minimum-
clearance primitive was on the trunk for 19, a front limb for 11, and a rear
limb for 23. At their calibration-selected sensor thresholds, 50 had support
in both depth and LiDAR and three in depth only. These support counts do not
imply discrimination: contact-negative candidates overlap the same clearance
range heavily.

Most importantly, physics-step replay found 141 branches with at least one
disallowed contact at 2 ms resolution, while the frozen 10 Hz H1 label marks
only 53. All 53 frozen positives had a physics contact, but 88 additional
branches contained transient between-sample contact and remained frozen-label
negative. There were 3,628 positive physics steps. This temporal-label
aliasing explains a substantial part of the full-geometry score overlap and
is why a purely geometric clearance state cannot reproduce the existing
sampled proxy as an oracle.

## Decision and next architecture

Because the privileged full-scene condition fails, sensor-limited conditions
cannot be interpreted as solutions. The required next specification is
`ARTICULATED_CONTACT_DYNAMICS_STATE_V1`, with explicit link motion,
controller/compliance response, candidate action, environment geometry, and
contact-interaction outputs. Before that experiment, the project should also
decide prospectively whether the proxy means any physics-step contact or only
contact present at the 10 Hz sampling instant. Another scalar contact head or
geometry predictor is not justified by this result.

The immutable aggregate reproduction ledger is stored outside the nearly full
workspace at
`/home/andrewknowles/.cache/lewm_go2_temporal_v03/h1_articulated_swept_geometry_sufficiency_v1/row_level_evidence_v1.npz`
(SHA-256 `827263fa58aaf782daddcca9c935173f46a0b4c44a672549cbc2daf8b4a7eea5`).

No model was trained; no learned checkpoint or JEPA predictor was executed;
no new identity was generated; and no memory, novelty, routing, or navigation
work was performed.
