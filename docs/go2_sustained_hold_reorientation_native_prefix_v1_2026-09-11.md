# Sustained-reorientation completion and native-prefix comparison

The completion helper admits only the original raw replay's exact frozen launch,
model, saved-input boundary, complete 407-observation stream and 404 compared
forecasts. It reconstructs each complete normalized comparison, verifies the
original saved row and candidate selection hashes, and recomputes the public
packet fingerprints from the actual RGB, depth, gyro and auxiliary-camera inputs.
The fingerprint ordering matches the sustained replay's exact input ordering.
No additional neural inference or training-ancestry replay occurs in this helper.
Its caller must separately authenticate the completed result and original inputs.

The native comparison requires 407 completed commands in both original and
successor tapes. Commands before observation 406 must be identical. At observation
406 the original hold changes to the candidate left turn. Every actual candidate
decision through that frame must match the completed prospective raw replay.

All 21,050 physical samples before the intervention must match exactly for every
recorded physics field, as must all 407 public sensor packets. Both physical
traces must additionally contain all 50 samples of the intervention command.
Those later samples may differ: their presence establishes a complete recorded
interval without asserting identical outcomes after different commands. The
complete successor raw audit remains required to validate its collected history.

Source: `scripts/sustained_hold_reorientation_native_prefix_development.py`.
Tests: `lewm/tests/test_sustained_hold_reorientation_native_prefix_development.py`.
This is a completion/comparison helper, not a native launcher or policy approval.
Preserve original failures and require completion of the existing native queue
before admitting any fresh sustained-turn native experiment.
