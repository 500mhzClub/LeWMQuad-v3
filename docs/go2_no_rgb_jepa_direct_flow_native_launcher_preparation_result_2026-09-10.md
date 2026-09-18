# Tracking-recovery native launcher prepared

The fixed one-episode launcher is implemented for the same no-RGB JEPA model,
maze-02 scene and DirectFlowResidualAnchoredController used in the pending
full-controller replay. It joins the prepared collector, full raw audit,
physical-prefix checker and original model admission. It adds no automatic
waiter and does not change the existing native queue.

The launcher requires the positive completed controller-prefix result, its
original process to have ended, and exact completion of the original six-case
batch followed by frontier, hold and contact. The queued experiments are
operational prerequisites; their policies are not adopted. The original model,
training and episode input admission is fully reexecuted. Queue completion and
artifact checks remain explicit, without claiming a final independent-population
policy review or rerunning all queued policies' training-input admissions.

The worker starts a fresh model/controller and a single fresh scene. A second
fresh model/controller performs the full raw audit. Collection, audit, physical
prefix and readout artifacts are retained if a later stage fails; partial
collection files remain on disk even if collection itself raises. Scientific
failure remains a valid audited result, while incomplete collection/audit/prefix
evidence yields a failed worker. Strict visibility and physical round-trip
criteria remain unchanged. A zero intervention request is not labeled movement.

Verification:

- 32 focused launcher/input/worker tests passed in 3.75 seconds.
- Earlier component checks passed: 53 native-prefix tests and six complete
  collector/audit source checks.
- Actual source preflight verified 2,120 bound paths, with no model loaded or
  runtime output created.
- Both actual live-owner gates rejected native admission: the controller replay
  owner and the original ordered native queue were still live.
- The replay had passed frame 700 at the last progress check; no completed
  positive boundary result was assumed.

The worker tests simulate failures at collection, audit, prefix comparison and
final verification. Input tests mock external completion/digest operations and
do not constitute actual positive runtime admission.

See [the launcher](../scripts/run_go2_no_rgb_jepa_direct_flow_maze02_pilot_v1.py),
[protocol](go2_no_rgb_jepa_direct_flow_maze02_pilot_v1_2026-09-10.md), and
[preparation record](go2_no_rgb_jepa_direct_flow_native_launcher_preparation_2026-09-10.json).
The native output root remains absent. Do not launch it until the completed
controller result is reviewed and the original batch/frontier/hold/contact
owners have finished and their results authenticate. No hardware or navigation
qualification is established by this preparation; the overall goal stays active.

Later in this turn, the full-controller replay reached frame 859 and reported
recovery through floor registration and planning. It selected `right_turn` with
request `[0.0, 0.0, -0.45]`, changing the original terminal zero request. All
859 preceding complete decisions and 856 original forecasts matched. A separate
receipt reconstruction checked all 860 saved comparisons and the bound original
episode and observer artifacts; its report is
[the in-progress boundary check](go2_no_rgb_jepa_direct_flow_controller_boundary_receipt_check_2026-09-10.json),
SHA-256 `59079f6a32183244ba46982294a48bd2f39921550085817babcea861cb9bde05`.

The controller report SHA-256 is
`77c335b55d3b0f225b26c925bf07b6ff9035d8cce9694582f599fe5614e639ff`,
and the closed decision stream SHA-256 is
`2748089cb1e2ea48661e44595c042bdbeb1d081d506e8379990eb5483d70bc50`.
These are intermediate artifacts, not a completed result: the same owner is
still reexecuting final full input admission. No new command was physically
executed and no native admission is claimed. Wait for that exact owner and its
terminal result, then authenticate the positive prefix and the original native
queue before launching the prepared pilot. The fifth matched native case had
reached observation 1977 at the last check.
