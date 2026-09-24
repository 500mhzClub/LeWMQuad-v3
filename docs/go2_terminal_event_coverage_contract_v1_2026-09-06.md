# Terminal-event acquisition coverage V1

This is a prospective contract for future fixed-inventory collection, with a
post-hoc compatibility check on all eight recorded corrected-pilot runs. It
does not replace the pilot protocol, repair its strict visibility failures,
convert partial command replays to complete replays, or grant training eligibility.

## Required separation

Each declared case stays in the planned denominator. An audited recording is
classified as schedule recorded, physical terminal recorded, pre-departure
physical terminal, setup failed, or infrastructure truncated. Missing/corrupt
recordings are invalid; never-attempted cases remain unattempted. A physical
terminal after action departure can be complete acquisition of that action's
observed outcome without completing the intended command schedule. These are
data properties, not successful navigation or calibrated safety probabilities.

The wrapper must run the frozen full raw sensor/contact/command/setup/first-stop
auditor before the new coverage layer. Native contact fields, attribution at
every physics sample, contiguous clocks, exact float64 requested commands,
all scheduled pre-terminal captures, and the final native sample are required.
The first physical stop must equal the last sample. Contact at an exact tick
boundary still interrupts that tick: the collector stops before its completion
flag and next image capture. No post-stop execution or images may be fabricated.

Setup and pre-departure failures produce no candidate-action targets. A recorded
contact establishes positive cumulative-event labels for later planned horizons,
but not later motion or images. A noncontact termination leaves later contact
outcomes unknown, not negative. Available earlier endpoints retain their labels;
missing or corrupted captures cannot be hidden by changing a validity mask.
This does not certify the finite-pixel depth measurement or deployment sensing.

## Bounded recorded compatibility check

`scripts/check_go2_terminal_event_coverage_v1.py` reads only the explicit artifacts
bound by the completed corrected eight-run pilot, plus its exact terminal audit.
It verifies the frozen 747-source predecessor, artifact hashes and commit byte
accounting before checking the new source bindings. It recomputes the full raw
audit for every run and requires report/prefix identity with its saved precheck.
It writes only new coverage metadata in the distinct exclusive child
`go2_terminal_event_coverage_check_v1_attempt_001` of the existing development
artifact root. Maximum new metadata is 32 MiB, with 40 GiB free-space reserve.
No physics, camera rendering, training, checkpoint access or previous-output
modification is performed. Any failure preserves the completed rows and terminates
this check; no silent case replacement or successful-subset denominator.

Synthetic tests cover complete schedules, contact and noncontact termination,
exact-boundary contact, pre-departure stops, rejected setups, infrastructure
truncation, missing terminal/native-contact/camera data, changed commands,
post-stop samples, wrong terminal reason, missing prefixes, corrupt censoring
and full-population missingness. Mocked accounting is not raw contact validation;
the recorded wrapper enforces the latter independently.

## Remaining prerequisites and scientific objective

The finite-pixel measurement/uncertainty assay remains required before freezing
fresh independent-layout collection. Keep the 12-layout 6/3/3 roles and planned
action/history/support/context coverage. Then test matched direct, supervised
rollout and JEPA predictors against action/time and zero-motion controls with
paired seeds and RGB/history/action ablations. Dependable local physical
execution, online rollout and memory/backtracking benefit, novel-maze missions,
real-time deployment-valid sensing and bounded hardware evidence remain unmet.
