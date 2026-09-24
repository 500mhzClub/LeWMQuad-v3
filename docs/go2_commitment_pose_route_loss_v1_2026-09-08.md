# Commitment-pose route-loss diagnostic V1

Use only the exact complete four-case native result and readout. Reconstruct
the two RGB-direct maps from their original public packets and already raw-
audited visual-pose evidence. This does not reestimate the observer or execute
a controller. Account for all actual rows, replay every available map receipt,
and retain unavailable/failure/drain rows in the denominator. Compare every
recomputed receipt, selected floor proposal and six candidate surface checks
exactly against the original record.

At each actual prediction context, record the measured map position, closed
start cells, their intersection with the original nominal obstacle inflation,
and observed-floor entry candidates within the original 1.25-m connector limit.
A blocked start cell belongs to every closed connector, so it proves that
additional views alone cannot remove that conflict from this monotone map. It
does not prove actual robot collision or justify removing the inflation.

Retain all conflicting articulated shape IDs, original voxel/witness identities,
candidate utilities and current-posture conflicts. Transform each first-hit
voxel's eight corners into the measured floor frame and report its height
interval and whether the measured floor plane crosses it. This is only a height
diagnostic: it does not identify the observed surface as ground, approve foot
support, waive a calf/foot conflict or change the frozen filter. Exact point/
surface classification would be needed for any later terrain-support treatment.

Inspect current hardware, require 8 GiB available RAM and 1 GiB planned output
above the 40 GiB artifact reserve. Benchmark independent one-/two-thread map
replays on the same first eight frames per case; require exact result equality
and select the faster measured width. Record the measurements and final hardware
state. These are bounded readout tasks, not native timing measurements or OS
quotas. Reauthenticate every source/input/URDF/artifact binding before and after.
Use exclusive root `go2_commitment_pose_route_loss_v1_attempt_001`; preserve any
failure. No simulator, training, action change or navigation qualification.
