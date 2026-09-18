# Extended-budget return-trip floor rejection

The persisted raw packets reproduce the stop at observation 3837. The required
normal translation correction changes from -0.04992621040152959 m at accepted
frame 3836 to -0.0502130758837221 m at rejected frame 3837. The latter exceeds
the unchanged 0.05 m development rejection gate. Normal alignment is
0.01933183284063218 rad, below the unchanged 0.10 rad gate.

Both cameras contribute to the reconstructed terminal plane: 5,462 primary
and 18,213 auxiliary candidates. Their maximum plane residuals are approximately
0.0122 mm and 0.0118 mm, respectively; the existing plane-validation checks
pass. The original visual observer has no terminal failure at either selected
late frame. Initial and last-accepted planes and corrections reconstruct
exactly; the original registration function reproduces the terminal error.

This establishes a height disagreement between the accumulated visual pose and
the measured floor under the static flat-floor hypothesis. These three frames
do not establish the history or physical source of that disagreement. The
slightly exceeded gate does not justify increasing it or resuming the failed
controller. The queued chained-anchor experiment addresses missing visual
associations and has not established a remedy for this height disagreement.

Evidence: `go2_extended_budget_floor_boundary_provisional_2026-09-11.json`,
SHA-256 `cd40fa9b1b726708c32db1d1bfcdaf6b31d524c0da14238b2b9402f4479fc61a`.
The diagnostic binds 2,142 sources and the exact consumed artifact files before
and after reconstruction. It decodes only three decisions and uses the original
extended-budget raw readers and floor geometry. No visual inference, model
inference, native execution, ground-truth pose input, or controller change occurs.

This is a provisional diagnosis of persisted data. The original native worker
was still running its post-collection checks. This receipt does not replace
native completion admission, the physical prefix comparison, or whole-run
sensor/model verification, and establishes no completed round trip or
navigation qualification.

The subsequent two-image probe reconstructs the original selected raw global
position and rotation exactly. The original pair has 177 lifted matches, 174
inliers, all 12 image grid cells represented, 1.335929 mm residual RMS, and
0.000278982 rad disagreement with the saved relative gyro rotation. All original
selected registration metrics and consecutive-pose gates reproduce.

The local transform disagrees with the two measured planes by -0.220000 mm
along the reference floor normal and 0.000191508 rad in normal alignment.
The accumulated global height correction changes by -0.286865 mm at this step;
it was already -49.926210 mm at the preceding frame. Thus the final crossing is
not a newly reconstructed gross correspondence failure. The history that led
to the preceding accumulated disagreement remains to be diagnosed. Both late
frames selected the primary camera and promoted a keyframe for half-feature
overlap; the terminal pose used frame 3836 as its reference.

Pair evidence: `go2_extended_budget_terminal_visual_pair_provisional_2026-09-11.json`,
SHA-256 `9796462c7206ebc8525b8e6b571097e12ee14da0c965eefb365c8aa045de73dd`.
It binds 2,143 sources and reuses the prior exact input bindings. The gyro values
come from the original bound visual witnesses; their full integration history
is not rerun. This remains a local raw-image reconstruction, not full observer
or native completion admission.
