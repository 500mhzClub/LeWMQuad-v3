# Completed hold-reorientation maze02 diagnostic

The intervention executed, but repeated reorientation did not solve the
navigation stall. The first changed command was an executed left turn instead
of hold at observation 405. The original comparison authenticated 406 common
observations, 21,000 physical prefix samples and 403 identical model forecasts.
All preceding requested commands and public/physical observations matched.

There were 138 hold-reorientation interventions, all left turns. Observed goal
distance was 3.353074 m at the first intervention, reached a subsequent minimum
of 3.348646 m at frame 468, and was 3.384232 m at the last admitted observation
1993. These distances describe the observed trajectory; they do not establish
the physical outcome of a different action or policy.

At observation 1994 the visual tracker still admitted a current joint pose.
Floor registration then rejected its required correction before planning. The
terminal receipt retains decision tick 1993 because the controller had not
admitted observation 1994. The subsequent ten observations are zero-command
drain. The distinction between observation frame and last admitted decision tick
is material when locating this failure.

Reconstructing the original measured floor planes from both raw depth cameras
at frames 0, 1993 and 1994 reproduced the initial and last accepted corrections
exactly, then reproduced the original rejection:

| Observation | Normal translation correction | Tilt correction | Result |
| --- | ---: | ---: | --- |
| 0 | 0 mm | 0 rad | Initial reference |
| 1993 | 49.816163 mm | 0.066193 rad | Accepted |
| 1994 | 50.016945 mm | 0.066463 rad | Height gate rejected |

The original limits are 50 mm and 0.10 rad. At the terminal observation the
combined plane fit passed, using 28 primary and 7,968 auxiliary candidates. Both
camera maximum residuals were below 0.0032 mm against the 3 mm coherence gate.
Thus the observed failure was the accumulated normal correction limit, not a
missing plane, incoherent plane, exceeded tilt gate, or lost visual pose. These
development gates are not calibrated physical pose-error bounds. The small
threshold exceedance is not a reason to relax the gate: the repeated-turn stall
already preceded it by many observations.

The complete run collected 2,005 observations and 100,950 physics samples. It
crossed two distinct open maze edges, with no invalid crossings, native contacts,
arrivals or round trips. Strict physical visibility passed. Median observation
and control time was 2,180.103 ms; every observation exceeded the 100 ms command
interval. This remains failed navigation and failed real-time qualification.

`scripts/diagnose_go2_completed_hold_reorientation_maze02_v1.py` completed with
exit zero in session 68967. It verified its 1,961-path source closure, reran the
original completion verifier, rehashed all 12,070 native artifacts, read all
2,005 decision rows in order, reconstructed the physical-contact readout, and
performed the three raw-depth floor reconstructions. It authenticated the
original raw audit, model identity and physical-prefix comparison without
rerunning visual tracking, neural inference, full raw audit or training ancestry.
The original native source closure contained 1,957 paths.

The machine-readable diagnostic is
`docs/go2_completed_hold_reorientation_maze02_diagnosis_2026-09-11.json`, SHA-256
`8b85fcb4a5c58706ad84604ebcfd5885d94b831ef9110e528d8a1a0991c02cf2`.
The native result SHA-256 is
`39e4e616361fde21d4cb5b6480794d3e062ace7bebfed04d70031e02a87937e9`.
The model remains
`35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`.

An initial read-only diagnostic failure is retained in
`docs/go2_completed_hold_reorientation_diagnostic_initial_failure_2026-09-11.json`:
saved JSON episode identities needed conversion from lists to the live
validator's tuple type. The corrected diagnostic restores only that container
type. No native attempt was restarted and no original artifact was changed.

The next policy review should address repeated hold/turn choices and pose
consistency over long histories separately. Complete the queued contact,
tracking and extended-budget diagnostics before choosing the independent-study
policy. This diagnosis does not approve the current policy or alter the frozen
queue, correction limits, success criteria, model, or observation contract.
