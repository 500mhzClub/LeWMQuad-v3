# Adjacent-pair correspondence stage diagnostic V1

Use the exact completed overlap-retention direct-039 native run. Reconstruct
all 230 available original evidence rows through its terminal frame 229 exactly.
For all 229 immediately adjacent 100-ms pairs, compare the frozen descriptor
matcher with the frozen gyro-seeded bidirectional flow helper on the same
corner-selected reference features. Run both on every pair, not only failures.

Count forward/backward descriptor ratio survivors, mutual and unique pairs,
bidirectional seed-consistent tracks and paired-depth survivors. A distinct
counter-instrumented copy must return arrays exactly equal to the original
matcher on every pair. Save both matchers' point/pixel arrays and all stage
counts. Flow uses current measured relative gyro only as its zero-translation
initialization; no command, future image, current pose or native seed enters it.

Pass both populations through the unchanged joint rigid fit, gyro consistency,
spatial support, residual/reprojection and current-increment gates. Retain every
failure and all qualified point masks, fits and registration receipts. The
reference is always the immediately preceding accepted original observation;
no candidate pair fit replaces or updates that observer. Save the full sensor
phase before opening native pose for separate pair translation/rotation errors.

The earlier whole-trajectory flow observers remain rejected. This narrower
diagnostic tests a different short-baseline population; it does not establish a
new observer, justify method selection per trajectory, weaken gates, calibrate
uncertainty or infer an unexecuted trajectory. Ten original terminal drain frames
are outside the fixed pair population because no new preceding observer pose
was accepted after frame 228.

Verify all source/input/output identities before and after. Bind the earlier
negative flow reports and latest exact-target result. Use an exclusive root with
terminal failure on error, 4 GiB RAM minimum, 40 GiB storage reserve and 256 MiB
output allowance. Record hardware before and after. One causal observer and 229
small pair comparisons run sequentially with one OpenCV thread. No training,
new physics/commands, original-result change, retry or navigation/hardware claim.
