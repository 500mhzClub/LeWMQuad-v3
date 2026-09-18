# Existing motion readout as a direct visual-servo component

Status: **COMPLETE**, CPU session 4257, exit 0. This independent 32.4-second
check used CPU cores 0-3 while the matched predictor job continued on GPU and
cores 8-11. It changed no weights and issued no commands. It evaluated current
and supplied-goal RGB at 10 fixed frames in each of the two exposed local tasks.
Physical targets were loaded only after visual predictions were complete.

The current 500-ms motion probe is poorly matched to this proposed use. Across
20 dependent states, mean displacement-vector error is 7.658 cm and mean wrapped
heading error is 9.020 degrees. At both initial states the actual goal is
27.70 cm away, but the readout estimates 6.60 and 5.93 cm; actual goal heading
is about 50.68 degrees, versus predicted 14.78 and 13.71 degrees. Near the goal,
it sometimes predicts forward displacement where the goal is behind the robot.
It misses four sampled within-tolerance states, with no false positive goal
classifications among these 20 states. This is not a closed-loop success test.

A separate training-support calculation (session 27417, exit 0) finds that its
3,518 training targets cover at most 10.52 cm planar motion. Signed forward
motion ranges from -0.846 to +10.520 cm; there are no targets below -2 cm.
Lateral displacement ranges from -1.752 to +1.383 cm. Heading changes range
from -13.592 to +13.310 degrees. These measured limits establish a coverage
mismatch for long and reversed goal displacement; they do not prove this is
the sole cause of every readout error or that visual geometry is absent.

Do not promote this unchanged component as a strong reactive comparator or
train on these exposed diagnostic images. A more suitable direct visual-servo
baseline should use the existing training-only 100-3,000-ms image pairs, with
both pair directions represented, signed body-frame goal displacement targets,
and the same frozen encoder. Match the data and fitting budget to the learned
cost as closely as possible and declare any target/capacity differences.
Prediction/planning is then compared with direct observed-image goal estimation,
rather than only an action-blind uniform-tie policy. This baseline is not yet
trained or executed; the active four-arm predictor comparison has priority.

Results: `go2_visual_servo_readout_diagnostic_2026-09-17.json`.
Source: `scripts/evaluate_go2_visual_servo_readout_development.py`.
Training-support values: `training_support.json` in
`/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1/go2_visual_servo_readout_diagnostic_v1_attempt_001`.
