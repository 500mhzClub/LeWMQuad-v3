# Hold and reacquire after temporary floor disagreement

The completed noisy matched cohort has learned 3/4 round trips and reactive
1/4, with zero contacts. Reactive layout 3 stopped at frame 405 because two
of 2,300 selected floor points exceeded the transported-normal residual limit
(maximum 3.018 mm versus 3.000 mm). Its exact 405-pose prefix and rejection
were reproduced without changing thresholds or using native state.

`lewm/floor_reacquisition_development.py` changes that geometric disagreement
to an explicit unavailable observation. It consumes the frame while preserving
the last accepted anchor and initial floor reference. The following camera
frame must pass the original packet, pose, floor and clock checks; other errors
retain their failure latch. No rejected pose is returned or sent to mapping,
motion correction or arrival measurement, and no floor threshold is widened.

The runtime cancels pending movement when the rejection is published and
requests hold while continuing sensor acquisition and tracking. Arrival dwell
and its previous-position/quiet-boundary state reset. The global mission budget
continues. Four consecutive newly accepted poses are required before planning
resumes, rebuilding the correction model's causal history. Plans computed from
pre-gap observations cannot be committed later. Existing map memory remains;
rejected measurements add no floor or free space.

Four focused tests pass: unavailable observations reset dwell and consume the
budget; only the specific floor conflict is recoverable; waiting and stale
in-flight plans cannot command motion; rejected frames never reach mapping or
pose history and four accepted frames clear the hold. A complete 409-frame
sensor replay passed all 405 original raw/registered pose-pair witnesses exactly
and treated frames 405–408 as unavailable. No subsequent recorded frame exists
to demonstrate reacquisition. Replay took 41.22 s on CPU 8 and used no physics
or controller execution. Evidence is in the original reactive-3 root under
`floor_reacquisition_replay_v1/result.json`.

Run one prospective native test using
`scripts/run_go2_floor_reacquisition_reactive_noise_development.py`, fixed layout
3 and the same stronger reactive policy, noise, cached tracker, mapper,
independent obstacle observer, actions, budget and physical guards. It changes
registration/mission missingness behavior only. Root:
`go2_floor_reacquisition_reactive_noise_2mm_native_layout03_4800_v1_attempt_001`.
Preserve the predecessor failure and every new outcome. This tests continuous
behavior; disappearance of an exception alone is not navigation success.

No native owners were active before launch; the prior replay has exited 0.
Use CPUs 8–15,24–31 with one numerical thread per process. Approximately
5.9 GiB is free on RecoveryStorage, enough for this single bounded recording.
The larger training-method comparison remains unstarted pending the result.

The prospective native trial launched successfully: PID 3560998 / session
83127, initialization log 03:36:06 local, with the required odd CPU group.
Actual launch metadata records ReacquiringFloorRegistration, four consecutive
accepted poses before resuming planning, dwell reset, command cancellation
and unchanged global budget. The inherited
`mapping_floor_acceptance_thresholds_changed=true` field describes the earlier
current-plane classifier change; it is not a new threshold change in this
reacquisition trial. Its recorded source uses the same mapper, floor candidates
and 3-mm residual limit as the immediate reference. No source used by the
running process was edited after launch.

A separate reviewed retirement of superseded training-control depth reclaimed
18.664 GiB while this run continued, leaving about 25 GiB free. All current
noisy comparison and debugging recordings remain fully retained.

## First native outcome: round trip, recovery branch unexercised

The owner exited 0 after 287.51 s including archive, max RSS 11,497,716 KiB,
zero swaps. Independent evaluation verifies goal frame 1096 and home frame
1897, maximum physical dwell distances 31.787/6.555 mm, quiet speeds
0.00821/0.01918 m/s and all dwell requests zero. Zero contacts or pipeline
faults. All 1,899 poses accepted, floor available 1,898/1,899, median/max pose
error 12.843/17.579 mm. Sampled path 20.186 m; selected/on-time plans 466/461.

There were zero floor rejection events and zero reacquisition holds. The
activation diagnostic is `floor_reacquisition_activation_diagnostic_v1.json`.
The recorded round trip is valid, but this trial does not demonstrate live
reacquisition or causally resolve the predecessor failure. The trajectory
changed despite the fixed noise recipe; do not select favorable survivors or
attribute the difference to a branch that never executed. All native jobs
have now exited.

Next: one explicitly declared floor-registration publication-gap trial on
reactive layout 3, withholding frames 405–408. This tests a four-frame loss
of accepted floor pose, with RGB/gyro tracking and independent depth sensing
continuing. It is fault injection, not calibrated hardware sensor noise.
Retain the last pre-gap floor anchor/reference, publish no pose for the gap,
require hold after rejection publication, reset dwell, and resume planning
only after four subsequent accepted poses. Verify actual command ledger and
pose/map/mission receipts, then independently evaluate the complete navigation
outcome. Do not silently add the injected-gap run to the natural-noise cohort.
A separate launcher is still to be implemented; no gap trial has started.
