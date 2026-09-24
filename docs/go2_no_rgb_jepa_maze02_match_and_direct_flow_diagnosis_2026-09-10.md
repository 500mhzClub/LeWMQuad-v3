# No-RGB JEPA maze-02 tracking failure and existing fallback

The completed no-RGB JEPA episode stopped at frame 859 after five distinct open
edge crossings, without contact or a verified arrival/round trip. The strict
visibility audit passed. The tracking diagnosis below does not change that
outcome or establish navigation recovery.

All 18 original camera/reference failures were reproduced: eight retained
anchors plus the immediately previous frame for each camera. Original
correspondence arrays were byte exact. The front-camera incremental pair had
four valid depth correspondences. The auxiliary pair had twelve, but the best
rigid consensus contained eleven; the minimum is twelve. Despite the error
label “insufficient rigid consensus after pruning”, this failure occurred
before the first pruning refit (zero rounds). Identity gyro was used only in
this bounded pre-gyro failure diagnostic; it was not a replay of the gyro gate.

The repository's existing direct corner-flow association was then applied to
the fixed 858→859 pair, without changing its source. It produced thirteen
front-camera and forty-five auxiliary-camera valid depth correspondences.
Repeated calls produced identical receipts and byte-identical arrays. A
separate verification authenticated 1,915 source bindings, all three output
artifacts and 5,260 original input bindings, and checked the saved finite array
shapes and count accounting. That verification did not independently rerun the
association algorithm. This probe did not fit or admit a pose.

The next experiment is the separately named observer prefix described in
[its protocol](go2_no_rgb_jepa_direct_flow_observer_prefix_v1_2026-09-10.md).
It starts both original and existing fallback observers at frame zero, uses
the actual recorded gyro history, checks the original raw visual receipts,
and stops at the first changed evidence or failure. Its boundary checks and
existing observer tests passed: 28 tests in 4.35 seconds. A recovered observer
pose still requires floor-registration and full-controller verification, then
prospective physical simulation; later observations from the original failed
episode cannot establish the outcome of a changed command.

Artifact identities:

- Match diagnosis result: `95b0315758c4065086bb11076a3e8f4767f1e660d15c51cb3ea2d72aa6a689a7`.
- Pair probe result: `4db83292904deee9bd35e10079e6a9dc424c0e8ec78567bf1829fbeb9621977b`.
- Pair verification: `3cecbe1165bf0bd56a4c81b7c9cba17b47a43d6924cc72f569c02b4f73c71ff8`.
- Observer-prefix launch: `7b66fee5a29cf7611e095175e20fceac44e2e439dcd8336b3f69accac77c1ea6`.

The overall goal remains active. The preceding status turn was a verified
wait: it polled the live profiling session and inspected the native worker.
This turn adds authenticated diagnostic evidence and executes the bounded
observer prefix; no sealed material, trained weights, live source closures,
native queue ordering or historical failure artifacts were changed.
