# Post-hoc tracking result: complete analysis, incomplete tracking and motion

The distinct analysis completed all eight raw audits, all 96 paired pose streams
and all 48 comparisons against the six explicitly exposed predecessor tapes.
The original V1 attempt remains terminally failed. This result completes section
9 of the new-thread handoff; it does not establish navigation or repair the
original experiment.

The continuity tracker maintains a complete pose stream on **6/8 tapes**, versus
**4/8** for the original tracker. Intended executed motion coverage remains
**0/8**. Strict depth visibility fails on three of 3,544 camera frames. No native
physical stop was found; a complete command schedule was never an observer-driven
mission and never proof that the intended displacement or turn was executed.

## Complete base population

Each tape contains 443 frames. Errors below are maxima over each candidate's
available population; missing frames have no invented error or pose.

| Scene / support / direction | Original available | Continuity available | Candidate max position mm | Candidate max orientation mrad |
| --- | ---: | ---: | ---: | ---: |
| Offset / nominal / left |147|443|4.350|0.764|
| Offset / nominal / right |343|443|5.272|0.622|
| Offset / low friction / left |443|443|4.902|0.521|
| Offset / low friction / right |443|443|3.657|0.513|
| Baffles / nominal / left |285|285|8.662|0.639|
| Baffles / nominal / right |443|443|5.807|0.601|
| Baffles / low friction / left |170|170|4.959|0.548|
| Baffles / low friction / right |443|443|3.520|0.480|

The two offset nominal interruptions are one-frame measured bridges at frames
147 and 343, with retained-anchor rejoin at 148 and 344. This adds 396 available
frames in total: 3,113/3,544 versus 2,717/3,544. On shared available frames, all
four scored error metrics agree exactly; the benefit here is continuity, not
improved shared-support accuracy. The unchanged empirical 20 mm / 2 degree
allocation is met on six complete candidate tapes and four complete original
tapes. It is not a calibrated bound.

Both estimators fail at frame 285 on nominal-left baffles and frame 170 on
low-friction-left baffles. Candidate terminal status is
`NO_CURRENT_MEASURED_TRANSLATION`: neither retained references nor the immediately
preceding frame supports a current accepted measurement. A longer bridge budget
cannot manufacture the missing increment. Both failed populations remain in
the results and denominators.

## Raw checks and representation

Raw sensor/contact reconstruction, actual geometry/setup, command selection,
stopping, native contact integrity, raster metadata, and all camera frames were
audited once per tape using the existing subordinate algorithms. These are
reexecuted production checks, not a separately implemented sensor/physics audit.

Only the copied raw adapter's function name/docstring and coverage callee differ
from the frozen `_raw_audit` function. A structural test verifies that exact
difference. Coverage applies the existing
`lewm.physical_execution_development.rotation_xyzw` acceptance convention to a
private normalized pose view. The independent numerical checker receives an
explicitly documented private view under that same convention; the original
scorer already uses SciPy normalized rotations. Neither raw arrays nor frozen
sources were modified. The original stricter gate failure is retained per tape.
Two calculations on this representation are independent arithmetic, not
independent acquisition of native truth.

Strict visibility failures are:

| Tape | Frame | Strict maximum residual m | Bad boundary rays | Bad stable interior rays |
| --- | ---: | ---: | ---: | ---: |
| Baffles / nominal / left |223|0.416028|1|0|
| Baffles / low friction / left |124|0.190226|1|0|
| Baffles / low friction / left |355|1.050446|1|0|

The fixed projected-footprint diagnostic passes its stable-interior metric on
all frames, with these three bad rays in its boundary-ambiguous population.
There are no near-plane occlusion failures in these three rows. Boundary pixels
are still uncertified and the strict result still fails. This does not prove a
specific renderer cause or a cause of the later tracker failures.

## All stress scenarios

Counts below are complete 443-frame pose streams out of eight tapes, not successful
missions. The machine-readable report retains all absolute/incremental metrics,
shared-support contrasts, failures, bridge/rejoin evidence and actual exposure.

| Scenario | Original complete | Continuity complete | Interpretation |
| --- | ---: | ---: | --- |
| Nominal wrapper |4|6|Matches base scientific outputs|
| One absent retained-anchor frame |0|6|Measured bridging helps; remaining failures retained|
| Ten absent retained-anchor frames |0|5|A ten-frame budget does not ensure ten valid increments|
| Eleven absent retained-anchor frames |0|0|No accepted eleventh bridge|
| RGB unavailable for one frame |0|0|Terminal current-sensor rejection|
| Depth unavailable for one frame |0|0|Terminal current-sensor rejection|
| Gyro unavailable for one frame |0|0|Terminal current-sensor rejection|
| Synthetic repeated RGB |0|0|Terminal rejection at onset|
| Depth drift |0|1|No complete tape meets the pose allocation|
| Shared gyro bias |0|0|Large accepted errors before terminal failure|
| Qualified increment conflict |4|0|Candidate stops on injected disagreement|

All eight tapes reach the intervention onset at frame 84 while both arms still
attempt updates. The exposure accounting distinguishes transformed sensor
packets from reference-call injections, and stops counting updates after each
arm fails. The original transforms/exposure admission is reused through exact
completed evidence and byte reauthentication; it was not rerun in this analysis.

Depth drift adds 2 mm per frame up to 40 mm. Maximum available position error is
65.141 mm. Shared body-z gyro bias is 0.02 rad/s, consistently applied by sample
timestamp to both gyro histories. Maximum available candidate position error is
287.510 mm and orientation error 90.140 mrad. Both arms share gyro-conditioned
rotation; agreement does not detect or correct this common error. Terminal
failure therefore does not establish a safe error bound before stopping.
These are fixed synthetic mechanisms, not calibrated hardware distributions.

## Timing, nonidentity and validation scope

Active-update timing includes the first failed update and excludes subsequent
cheap terminal no-ops. Base candidate median times range from 59.07 to 71.48 ms;
10 active updates exceed 100 ms and the maximum is 182.12 ms. Original medians
range from 43.83 to 49.89 ms, with two updates above 100 ms. These are saved
observer-only timings: acquisition, control, rendering and other full-loop work
are excluded. Physics did not continue during this observer computation.
Neither average nor tail observer timings qualify a real-time robot loop.

All 48 predecessor comparisons find nonidentical actual box inventory,
sufficiently different start, and different initial RGB/depth values. This is
not a topology-isomorphism test or statistical independence. There are two scene
clusters, two support settings and two turn directions; frames and stress
variants do not add independent layouts.

Focused validation: 39 adapter/precision/coverage tests, 141 existing numerical,
continuity, predecessor, raw-scoring and footprint tests, and five summary
accounting tests passed (**185 total**). No full-tree test discovery was used.
The 141-test job overlapped the analysis after its component benchmark; measured
analysis times include that competition and are not isolated throughput claims.

Hardware preflight found 16 physical / 32 logical CPUs, about 78 GiB available
RAM, and about 106 GiB free artifact storage. Equal four-tape component workloads
took 4.887/3.626/3.208 seconds at 1/2/4 workers with identical fingerprints.
Four workers were selected. Full per-tape work took 44.08–46.12 seconds; peak RSS
was at most 1,205,809,152 bytes. Each worker had a 16 GiB address-space limit and
bounded output writes; the total declared derived-output envelope was finite,
with a 5 GiB free-space reserve. The completed analysis took 115.75 seconds and
wrote 22,545,770 bytes before its terminal result. No GPU workload ran.

## Artifacts and immediate implementation consequence

The complete scientific report is
`docs/go2_tracking_posthoc_raw_accuracy_scientific_readout_2026-09-07.json`, SHA-256
`f7265972cfd3770e59bf14a8389838b1b14d14e34087fa12a2488f1cf30c5bf5`.
It binds the exact sources, original input identities and all new output files.

Derived root:
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_tracking_posthoc_raw_accuracy_v1_attempt_001`.
Terminal `result.json` SHA-256:
`99b0ed6c47dc0c974579bc145b9888e6e3edd64a69db633f929fd327ab526531`.
The main reader is `scripts/read_go2_tracking_posthoc_raw_accuracy_v1.py`;
the summary reader is `scripts/read_go2_tracking_posthoc_science_v1.py`.
Do not repeat either completed analysis to rediscover these results.

The new evidence separates three implementation needs: actual sensor-feedback
motion control, loss of current visual translation support, and undetected
correlated sensor error. The continuity change solves two short interruptions
but neither the two baffle failures nor gyro/depth drift. Do not increase its
bridge budget or relax its gates on these outcomes. Use a separate source
variant for any new experiment. The existing joint RGB-D solver is the bounded
candidate for an independent-rotation comparison; it must reach both reference
branches and retain incremental rotation witnesses. Its prior nominal errors
were worse, so adoption is conditional, not assumed. Sensor-feedback execution
must still test fresh translation, signed turns, braking and return, including
the visible low-friction failures. None of this substitutes for a learning task
whose goal-progressing action choice actually depends on the scene.

The long-term goal remains active and unachieved.
