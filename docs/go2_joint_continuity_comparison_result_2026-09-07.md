# Joint RGB-D continuity: completed development comparison

The candidate meets every continuation criterion fixed in
[the pre-run candidate record](go2_joint_temporal_anchor_candidate_2026-09-07.md).
It preserves nominal availability and removes the large accepted pose errors
seen under both signs of shared gyro bias on these exposed tapes. This supports
fresh development execution. It does not establish reliable tracking, controller
adoption, independent validation, or novel-maze navigation.

This completed result supersedes the pre-run record's historical statement that
replay had not been launched. That source-bound record remains unchanged.

The [complete scientific readout](go2_joint_continuity_comparison_scientific_readout_2026-09-07.json)
retains all 32 conditions: eight 443-frame tapes, each with nominal sensing,
positive and negative 0.02 rad/s shared body-z gyro bias, and a missing-current-RGB
frame. Perturbations begin at frame 84 and preserve overlapping fast/body gyro
samples by timestamp. Every arm reaches the intervention while active. Failure
is terminal; unavailable rows remain in the denominator.

| Condition | Gyro / joint complete tapes | Gyro / joint available frames, of 3,544 | Gyro / joint maximum accepted position error |
| --- | --- | --- | --- |
| Nominal | 6 / 6 of 8 | 3,113 / 3,116 | 8.662 / 17.926 mm |
| Positive gyro bias | 0 / 1 of 8 | 863 / 2,213 | 287.510 / 15.337 mm |
| Negative gyro bias | 0 / 1 of 8 | 865 / 2,210 | 250.159 / 15.337 mm |
| Missing RGB | 0 / 0 of 8 | 672 / 672 | 1.681 / 7.917 mm before failure |

The baseline is gyro **continuity**, not the older noncontinuity estimator.
The legacy scorer keys `original` and `temporal_anchor` mean gyro continuity and
joint continuity respectively in this experiment. Twenty-four historical gyro
runs were reused only after exact source, packet-transform and stream checks.
The eight negative-bias gyro runs and all 32 joint runs were newly inferred.

Every available nominal and biased joint pose stays within the existing
empirical 20 mm / 2 degree allocation. Joint maximum orientation error is
9.470 mrad in these conditions. Positive-bias gyro poses exceed the position
allocation on 163 accepted frames and the orientation allocation on 52;
negative-bias counts are 164 and 53. The gyro maxima are 90.140 and 75.786 mrad.
Missing RGB remains terminal at frame 84 on every tape for both arms.

The paired analysis subtracts each arm's nominal error at the same frame from
its biased error, using post-onset frames available in all four relevant
streams. Joint bias-induced error is exactly zero on that common support for
both signs; gyro error increases. This support is only 15–46 frames per tape
after onset because the gyro baseline stops early. The paired result therefore
does not establish whole-tape bias robustness. No gyro bias was estimated or
calibrated.

Nominal accuracy has a cost: the joint maximum position error worsens on seven
of eight tapes and orientation error worsens on all eight. The worst position
error nearly doubles while remaining inside the fixed allocation. That
allocation is empirical, not a certified error bound. Nominal availability
increases by three frames on lower-friction left baffles; no tape loses nominal
availability.

Only nominal-left offset-niche completes under either bias sign. Five additional
biased failures per sign exhaust the unchanged ten-frame measured bridge after
retained visual references fail the existing gyro rotation consistency gate.
Saved witnesses show a current incremental fit remains available at these
bridge-limit failures. Primary reference failures report image/gyro rotation
disagreement; other references also encounter matching or consensus failures.
This localizes a remaining limitation of the uncorrected gyro reference check;
it is not evidence to increase the bridge budget or relax acceptance gates.

The nominal-left baffles tape still loses current measured translation at frame
285. Lower-friction left baffles loses it at frame 173, versus frame 170 for
gyro continuity. Both failures also persist under each bias sign. Their saved
incremental reason combines rigid consensus fraction, grid support and
displacement rejection; it does not identify which individual threshold failed.
No precise sub-cause is inferred from that combined label.

Independent verification reconstructs continuity history, selected rotation
witnesses, reference-to-current matrix products, gyro disagreement, and selected
pose agreement from sensor-only streams. It does not independently refit image
correspondences. All 32 complete sensor streams were admitted before any
evaluator-only native pose reached the scorer. Numerical errors were separately
reconstructed using the explicit private quaternion-normalization interface.
Original inputs, failed terminal records and strict raw-audit outcomes remain
unchanged. In particular, three strict depth visibility failures and zero of
eight intended-motion coverage passes remain. Two scene clusters with support
and direction variants do not constitute eight independent layouts.

The runner is `scripts/run_go2_joint_continuity_comparison_v1.py`; the independent
history and rotation readers are
`lewm/joint_continuity_history_verification_development.py` and
`lewm/joint_rotation_witness_verification_development.py`. The compact scientific
reader is `scripts/read_go2_joint_continuity_comparison_science_v1.py`.
The candidate/comparison focused tests passed 23 tests in 6.51 s; four additional
scientific-summary tests passed in 1.76 s. The latter ran during final admission
and scoring, not as an isolated throughput measurement.

Before execution, the machine had 16 physical / 32 logical CPUs, about 77.3 GiB
available RAM and 105.8 GiB free artifact storage. The GPU was not used.
Identical bounded four-tape, twelve-frame benchmark fingerprints took 6.195 s,
4.412 s and 3.703 s at one, two and four workers. Four workers were selected;
peak worker RSS was 1,259,560,960 bytes. The entire experiment took 315.583 s
and wrote 107,684,398 bytes before its terminal result, with 99 bound artifacts
and 851 source bindings.

Joint nominal per-tape active-update medians range from 70.06 to 84.17 ms,
with a 207.25 ms maximum and 12 updates above 100 ms. Historical nominal gyro
medians range from 59.07 to 71.48 ms, with a 182.12 ms maximum and 10 updates
above 100 ms. These are not contemporaneous paired speed measurements.
Acquisition, control and copying are excluded, and physics did not continue
during observer computation. A deployment-valid 10 Hz loop is not demonstrated.

The exact terminal root is
`/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_joint_continuity_comparison_v1_attempt_001`.
Its `result.json` SHA-256 is
`2808f8988fa0c9a04ddb1207c80a71398ec4f3d35f556bd77d0a6dbc89becb6b`.
The scientific readout SHA-256 is
`5945d0fbf985dfd45f62202a59d546c1375ab6d7439316d653cd4a0c53b8cda3`.
The readout binds all experiment artifacts and sources, and records the
predecessor result identity. These bindings grant no source-export authority.

The next useful step is a distinct, bounded fresh sensor-feedback execution
experiment using the existing empirical pulse planner as a baseline. Its pose
contract must explicitly accept verified joint rotation; existing gyro-only
interfaces must not be bypassed by relabeling evidence. Preserve current arrival
and stop criteria, measure complete-loop timing and realized motion, and retain
nominal and low-friction failures. Existing `InnerGoalPulseServo` already replans
from current observations and records action-response residuals; it uses a fixed
table and has no online adaptation. Reimplementing that functionality or
rerunning these completed tapes would not resolve the execution question.
The long-term goal remains active: learned scene-dependent action selection,
physical exploration/backtracking, independent maze success and realistic
deployment evidence are still unestablished.
