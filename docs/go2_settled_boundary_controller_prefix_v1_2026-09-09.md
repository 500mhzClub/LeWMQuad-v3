# Full raw-controller prefix for measured settling with a quiet boundary

The completed saved-mission comparison bf875306754c620d393d7a05496081c318c17d624726d2ae8bf348bab9caa053
reproduced all1867 original mission receipts through1866. Its first interval
counter difference was1857 (observed motion0.108114379705m/s); at1866 its
candidate had only9 quiet intervals. A further separately named successor
requires the first low-motion observation to establish the start boundary,
then counts ten complete subsequent intervals whose current and previous
observed motion estimates are at most0.05m/s. This prevents a window from
starting before the first observed quiet boundary. Interval averages still
cannot certify continuous native speed, so qualification remains unchanged.

Use SettledBoundaryRoundTripController, retaining all original tracking,
floor registration/map/contact resolution, model weights, action forecasts,
selector, residual memory and shared mission budget. The failed subpixel
candidate remains excluded. Six boundary tests and the eleven predecessor
mission/controller tests cover the implementation and exact source derivative.

Run one complete fresh raw public-sensor/controller replay on the ninth saved
collection, comparing each entire normalized decision with its saved original.
Allow only declared settling fields, quiet counters and mission phase/goal/
arrival fields to differ. Every original observer, map, model forecast, contact
query, action selection and actual requested command must remain exact.
Stop at the first mission behavior change, no later than1866, without reading
the next decision or applying any new command. Save all candidate decisions.
The saved-mission result supplies this bounded hypothesis, not native success.

The ninth native audit may run concurrently. Its completion is required before
using this replay to launch a new native episode; this prefix does not replace
its full physical/sensor/command/visibility audit. Bind the native launch and
collection identities, original full compressed decision stream, paired public
RGB-D/auxiliary files for frames0..1866, complete required public manifests and
histories, executed command tape, model admission and the saved-mission result.
Recheck source/input/model identities before and after. No training, simulator,
new acquisition, original-attempt restart or source mutation.

One CPU replay process and one numerical/OpenCV/Torch thread beside the existing
single native audit. Inspect topology, affinity, utilization, competing jobs,
GPU/VRAM, RAM and storage immediately before launch. Require8GiB available RAM
and1GiB output above40GiB reserve; record allowances as admission checks, not
OS limits. Exclusive output go2_settled_boundary_controller_prefix_v1_attempt_001.
No full-loop speedup, verified arrival/return, independent-layout generalization,
strict visibility, hardware or overall-goal completion claim follows from this
prefix alone. Preserve every outcome.
