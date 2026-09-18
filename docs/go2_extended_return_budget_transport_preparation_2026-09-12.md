# Extended return budget: measured floor transport component tested

`lewm/extended_return_budget_transport_development.py`, SHA-256
`04262b695ca404a8cdd06155950a69ea82cd8f8c619bd766a4e212748b6dc06d`,
extends the measured-floor transport composition's current-frame ceiling from
4,096 to the prospective complete population of 8,014 observations. The
composition body is otherwise identical to the original. Its pose accessor,
transport-evidence producer and tiled-density registration method use private
bindings to that composition. Original functions and globals are unchanged.

The limit comes from
`lewm/extended_return_budget_mission_development.py`: 8,000 navigation steps,
three warmup commands, ten terminal drain commands and one initial observation.
The new transport path accepts a properly reconstructed witness at frame 8013
and rejects frame 8014. This change concerns the transport-during-missingness
branch. Fully available floor registration retains its original accessor;
the complete controller and recording population still need their own bounds.

No plane-availability rule, measured residual threshold, rotation composition,
correction-magnitude constraint, calibration, initial floor reference, original
visual witness or qualification flag changes. Current transport must still
reconstruct from an earlier admitted floor pose and a current qualified visual
pose. Geometric conflict cannot be relabelled as missingness agreement.

## Verification and preserved initial test failure

`lewm/tests/test_extended_return_budget_transport_development.py`, SHA-256
`e4244b061edb7316d093a2651b2ff74c028e90430cdf16511f0af80520cf7232`,
passed **22 tests in 7.23 seconds**, tool session 23592. It covers:

- Complete transport reconstruction at frames 2, 4095, 4096 and 8013, with
  exact old/new evidence equality in the original admitted range and rejection
  by the unchanged original function beyond its ceiling.
- Rejection at frames 8014 and 8192.
- Late-frame rejection of forged poses, anchors, raw witnesses, missing-plane
  claims, residuals, counts, camera hashes, clocks, identities, qualification
  claims, corrections, future anchors and rewritten initial references.
- Rejection of current geometric conflict at the new final frame.
- Exact registration receipts and state through an available-plane, missingness
  and reacquisition sequence, followed by duplicate-observation failure latching.
- An AST comparison proving the composition transcription changes only the
  declared frame ceiling.

These are synthetic witnesses. Late frames are formed by shifting current and
reference witness clocks/indices while preserving the original frame-zero
floor reference. They exercise the real pose accessors, plane checks and
composition code, but do not claim a complete intervening raw visual history
or independently verified image fitting. A full actual controller prefix and
fresh native execution remain necessary.

The first test run, tool session 89212, had **four failures and 18 passes in
7.24 seconds**. All four failures were an incorrect test expectation of exact
identity rotation. The rendered fixture's measured plane correction has a
maximum off-identity element of approximately 1.342072498126718e-9. The test
was corrected to compare exactly against the rotation composed from the
measured anchor, raw-anchor and current poses. No implementation or numerical
tolerance was changed to resolve those failures. Both runs used the original
single-thread environment and `pytest -q -p no:cacheprovider`.

## Integration findings and remaining work

A separate fresh-controller constructor inspection, tool session 6575,
confirmed the currently composed objects: `MeasuredFloorTransportMission`,
`BodyProjectedTiledFloorMap`, `MeasuredFloorTransportMemory`,
`MeasuredFloorTransportResidual`, `ReceiptCopiedFootprintSelector`,
`TiledDensityFloorRegistration` and `MeasuredPlaneChainedAnchorVisualMotion`.
The eight persistent bound indices are `SinglePassMeasuredSampleBoundsIndex`;
both retained patch histories are `ProgressiveBatchedRetainedFloorPatches`.
The controller was constructed with `None` model/geometry only for this
inspection: no model was loaded, observed, called or trained, and no scene was
created. Its tick was -1, route empty and observation contexts inactive.

The selector and receipt-view path have separate exact memory-type checks in
`lewm/fused_scoped_batched_controller_development.py` and
`lewm/receipt_copied_footprint_development.py`. A future memory derivative needs
consistent private bindings for both. Updating only the selector check would
still leave the receipt view rejecting the new memory type. These paths must
preserve their optimized receipt behavior and all observed state aliases.

Next implement the larger bounded surface route, later floor evidence and
body-projected patch histories, then connect the new pose accessor to memory,
residual and mission advancement. Recording, packet/replay population and
evaluator bounds also remain incomplete. The broader integration map is
`docs/go2_extended_return_budget_preparation_2026-09-12.md`.

The active 4,014-observation chained/single-pass comparison was not changed.
All 2,639 source bindings were independently rehashed afterward and matched;
the two new transport files are outside its source roster. The short tests
and constructor inspection overlapped that already declared non-isolated
timing run. No extended native trial, runtime adoption or new navigation
qualification is established by this preparation.
