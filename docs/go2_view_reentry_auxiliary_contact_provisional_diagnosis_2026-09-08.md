# Provisional auxiliary contact diagnosis after recovery collection

The native recovery job is still running its independent raw audit. This note
describes recorded public decision and depth evidence only; it does not add a
completed native result, verified arrival or physical contact classification.

Root: `go2_view_reentry_maze_pilot_v1_attempt_001`.
Completed decision stream SHA-256:
`d05990c76efe733399cd3a7049ddec33db1bec65fc93299c23b8e29c9c639b7a`.
The stream hash was unchanged before and after the diagnostic. The public
auxiliary packet helper checked each cited depth array against its acquisition
hash and reconstructed its exact public validity mask.

The last selection at tick 1507 had no action. Its observed outbound goal
distance was 1.2742285923386731 m, with no arrival. The final drain record is
tick 1517. All six candidates passed all eight nominal path segments and their
primary surface checks. Every candidate failed its auxiliary surface check,
with its first conflict in the front-left nominal foot sphere. Current nominal
clearance was 0.5275333319814394 m and the observed route reached the goal cell.
Thus the terminal constraint evidence differs from the earlier nominal-radius
and view-phase failures.

`scripts/view_reentry_auxiliary_floor_diagnosis_development.py` reconstructed
the cited latest contributing frames using only public depth and recorded
observed poses. Whole-frame counts of floor/other returns matched each recorded
auxiliary receipt exactly. It then split the original floor classification into
the four neighboring mesh-quad checks and the nine-pixel height-band checks.

| Latest frame | First conflicting voxel | Reconstructed other samples | Nine-pixel height residuals above retained plane |
| --- | --- | --- | --- |
| 1420 | [148, -3, -10] | 5 | 0.010534951733–0.010575838013 m |
| 1424 | [148, -4, -10] | 2 | 0.010711466844–0.010729884230 m |

All seven cited samples had valid nine-pixel neighborhoods and passed all four
ground-geometry mesh checks. Each failed the original 0.01 m fixed-plane height
band. No class, threshold, map point, candidate or command was changed. This
covers the latest contributing frame of each first conflicting voxel, not all
historical samples or all possible contact voxels.

This supports investigating the registration of measured floor to the fixed
initial plane. It does not establish physical floor truth or justify exempting
unknown contacts. The mapper fixes its up axis from the initial public specific
force mean and its floor height from the initial measured floor median; both
remain fixed. A prospective investigation should separate initial plane tilt,
later observed-pose error and measured surface variation using public geometry,
with native information restricted to independent evaluation. Do not widen the
band or relabel these samples based on this partial witness check. First finish
the native audit/readout, bind this stream to that completed result, and preserve
the failure.

## Initial versus contemporary measured-plane follow-up

Separate public-only least-squares fits compared initial and contemporary
primary/auxiliary depth candidates. The initial fits use measured ground-mesh
candidates within the original 10 mm height band; contemporary fits use the
ground-mesh candidates without that band. No fit is a policy admission or a
physical-floor certificate. Source:
`scripts/view_reentry_floor_plane_registration_diagnosis_development.py`.

Initial primary/auxiliary plane tilt from the retained up axis is only
0.000509514/0.000534361 rad. Using these measured initial planes leaves the cited
later samples about 11.40–11.66 mm away. Correcting only the initial plane is
therefore not supported as a resolution of this recorded mismatch.

| Fit frame | Sensor | Measured candidates | Plane RMS residual | Same-frame cited sample residual |
| --- | --- | --- | --- | --- |
| 0 | primary | 108,114 | 0.105622 mm | not applicable |
| 0 | auxiliary | 305,323 | 0.014591 mm | not applicable |
| 1420 | primary | 66,723 | 0.030253 mm | 0.012969–0.014348 mm |
| 1420 | auxiliary | 263,017 | 0.012574 mm | 0.000226–0.001664 mm |
| 1424 | primary | 68,345 | 0.031504 mm | 0.021434–0.021773 mm |
| 1424 | auxiliary | 264,790 | 0.013003 mm | 0.001628–0.001955 mm |

The contemporary primary fits contain no auxiliary samples. Their agreement
with the disputed auxiliary samples supports a common observed floor-plane
registration mismatch. It still does not distinguish physical surface truth
from all possible shared observation/pose errors. The next investigation should
consider continuous observation-based plane registration, preserve measured
surface variation and unknown contacts, and use native pose only for independent
evaluation. No correction coefficients may be fitted to native truth.

Two focused plane-fit checks passed (known tilted plane plus nonplanar residual,
and degenerate/nonfinite rejection). A fresh consolidated helper invocation
reproduced all six fit populations and RMS values with the decision stream hash
unchanged. This diagnosis is now included in the unlaunched completed recovery
readout; its source files were confirmed absent from all applicable frozen
launched source bindings before modifying that readout.
