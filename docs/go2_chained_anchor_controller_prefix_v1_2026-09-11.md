# Chained retained-anchor full-controller prefix V1

Use the completed original no-RGB JEPA direct-flow native worker and the verified
chained-anchor observer prefix. Reload the original assigned model through its
existing snapshot/coefficient verifier and require state SHA-256
`fb6f1aba8830a53d67cd6c284fb24199966d5f0c63db3b2a107ab833c81c266f`.
Reuse the worker's completed model admission; do not retrain or re-evaluate the
training dataset. This distinction must be explicit in the launch record.

Construct a fresh `ChainedAnchorResidualController`. Its inherited map, floor
registration, mission, residual, action selector, model interface and 3000-tick
budget remain unchanged. Only the observer and declared result identity change.
Replay frames 0–853 using the original public RGB-D/body/gyro packets. Compare
all 853 earlier complete decisions and their 850 learned forecasts exactly,
apart from the new controller name and explicit capability flag. Require the
complete raw visual evidence to equal the independently verified observer replay
at every consumed frame. Reconstruct actual original command endpoints and
check input fingerprints and array preservation.

At frame 853, require the exact original measured bridge and authenticated
retained-anchor reacquisition. Check the full controller's floor-registered
current pose and forecast/action consistency. Preserve a terminal zero command
if a downstream controller gate fails. Report whether the controller admits
the reacquired pose and whether its requested command changes. The original
controller was live at this boundary; do not call this recovery from an original
controller terminal failure, and do not infer navigation recovery.

Stop at that boundary even when both requested commands are identical. Consume
no subsequent observation. Execute no new command. Verify unchanged model state
and absence of gradients afterward, plus all bound worker/observer artifacts
and sources. Preserve every failed result under the exclusive attempt root;
no automatic retry or replacement.

Run after both prior full CPU owners have ended, with one CPU worker and one
OpenCV/BLAS thread. Assess resources first: 16 GiB replay plus 32 GiB concurrent
native RAM reserve, at least 41 GiB artifact free space, and at most 1 GiB of
derived decision receipts. This is an offline learned-controller comparison;
native simulation, independent-layout studies, real-time qualification and
hardware evidence remain separate future work.
