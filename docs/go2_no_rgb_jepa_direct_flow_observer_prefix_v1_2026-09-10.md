# No-RGB JEPA direct-flow observer prefix V1

Replay the completed no-RGB JEPA maze-02 recorded sensor stream from frame zero
through the first changed observer evidence, either observer failure, or frame
859 inclusive. Use the existing DualCameraVisualMotion and existing
DirectFlowDualCameraVisualMotion in independent instances, with the actual
recorded front/auxiliary RGB-D and uninterrupted fast gyro history. All
baseline visual receipts must reproduce the original recording exactly after
JSON tuple/list serialization. Normalize only the added fallback receipt when
comparing candidate evidence. Stop before reading another observation once
pose/evidence changes; no counterfactual commanded future is inferred.

The completed pair-probe result is SHA-256 bound in the launcher. Its 5,260
original input bindings and inherited source closure are verified before and
after replay. New launcher, boundary tests and this protocol join that closure.
Use an exclusive new attempt; preserve failures without retry or resume. Public
packet fingerprints are checked before and after each observer. Current poses
must pass the original public pose accessor, including camera and sensor
identity checks. The fallback retains the original rigid, gyro, temporal,
displacement and bridge checks; it changes association only when the complete
original two-camera measurement policy reports missingness.

Run one CPU observer replay, OpenCV/BLAS one thread, with at least 40 GiB RAM
available (8 GiB observer plus 32 GiB concurrent native allowance), and 40 GiB
artifact reserve plus a 256 MiB output allowance. It creates no simulator scene
and loads no learned model. The existing native queue remains its sole owner.

This is an observer diagnostic. It does not replay floor registration, mapping,
the learned planner, command selection or physical execution. Even an accepted
boundary pose requires a subsequent full-controller causal prefix and a fresh
prospective episode before any navigation improvement can be claimed. Existing
scientific failures and the zero-round-trip result remain unchanged.
