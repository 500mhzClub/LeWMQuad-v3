# Measured-line / integral navigation V1: audited result and next decision

Collection62392 completed both fixed development missions, exit0. Full audit41535
passed both, replaying772 controller decisions and782 RGB/depth/relative-state
observations. Prelaunch tests37363 passed1,121 across101 files in76.36 s. Preflight
51452 and terminal audit verified290 source,201 input,2 gait and10 native bindings;
an additional post-audit checksum check also passed. All launched files are frozen.

## Physical result: still zero successful missions

Both layouts now complete one measured local arrival at tick179, acquire a scan,
select a branch at tick249, and enter alignment at tick265. Both terminate with
FAILED_ALIGNMENT at tick385 after the unchanged12-s alignment budget; total
elapsed time including release is39.0 s. Neither discovers the marker or returns
home. Neither records contact, native body stop, sensor fault or false home claim.
Zero-command release passes. Trusted graph edges remain zero.

North path length is1.755466 m and final home distance1.530640 m. South path
length is1.770481 m and final home distance1.531437 m. North arrival is0.04204 m
short of its target with933/933 nominal volume samples supported and no conflicts.
These are development observations, not continuous-volume/future-gait safety or
recognized-place certificates. More reliable local execution has not yet produced
whole-maze success, memory benefit or JEPA navigation evidence.

## Measurement evidence

Both runs retain full translation estimates on390/390 intervals. Maximum step
errors are0.77640 mm north and0.95351 mm south; final relative-position errors
against evaluation-only physics are0.45083 and0.67223 mm. All391 depth-frame
checks pass in each layout. The previous south rank loss and four north depth
failures remain recorded in their original experiment; these new trajectories
do not retroactively qualify old data or establish general sensor robustness.

The line/integral intervention changed two control components together. It
avoided the prior south failure on this run, but the two-layout population does
not isolate either component's contribution or establish independent generality.

## Alignment failure mechanism exposed by actual execution

The new integral controller reaches the original0.02-rad heading tolerance:
north at ticks290,316,331,352,365,376; south at287,288,314,329,352,364,374.
At those observations projected heading rate also satisfies0.1 rad/s. However,
the controller immediately resets the integral and requests zero on entering
tolerance. Neither trial maintains the required0.3-s dwell.

North's first entry has error0.01877856 rad and rate0.03815481 rad/s. After one
zero-command interval, error becomes0.02012716 and rate-0.03254466; by tick294
error is0.02638704. Subsequent cycles reenter near the threshold and leave it
again. Terminal errors are0.02120284 north and0.02016945 south. These physical
records show a release/settling control problem, not evidence that accepting a
slightly larger tolerance would meet the original criterion. Synthetic deadzone
tests did not model this release transient and therefore could not establish
actual alignment success.

## Next implementation, before a fresh whole-task experiment

1. Add a distinct release-aware alignment operator. Keep bounded feedback active
   toward the centre of the acceptance region, rather than releasing at its edge.
   Use an explicitly tighter inner control target, then a separate zero-command
   settling phase judged from fresh gyro/heading measurements. Preserve the
   original outer0.02-rad/rate acceptance, required dwell and12-s total deadline.
   If settling fails, bounded correction may continue within that same budget;
   do not complete from a pre-release observation or reset the deadline.
2. Add synthetic release-transient, delayed response, overshoot, no-response and
   bad-clock regressions. Require terminal zero and prove the integration does
   not replace active history. A tighter inner control target is not a relaxed
   success criterion; it still needs actual closed-loop validation.
3. Freeze a new named source/protocol and run both original full missions with
   line guidance, sensing, clearance, marker/memory and physical scoring retained.
   Replay every command and sensor sample. Preserve failures rather than edit
   either completed controller or extend its run.
4. If independent motion information becomes necessary later, implement a causal
   deployment-valid sensor-fusion successor with explicit uncertainty. The present
   rank test and stop on unobserved translation remain; neither commanded zero
   nor a low registration residual fills an unobserved direction.

Reliable discovery/return must precede matched memory, supervised/JEPA and genuine
multi-step planning comparisons, independent layouts/seeds, sensor robustness and
bounded real Go2 evidence. The ultimate scientific objective remains active.

## Exact identities

Output: `.generated/go2_measured_line_integral_navigation_development_v1_attempt_001`.

- launch.json: `703fbd6f2c86ab59e0e428214de3ec1094e93f180831ff82dc45e1eed7994bff`
- result.json: `e43b5b9bb7b16d60073cd5ea4b50dd385f9351d649db7273f5cba5edac5455d2`
- raw_artifact_audit.json: `6c5503e6abbe92f4a8b7fbf60ff844086def80545dd0b116d1e5dd1dbb05107b`
