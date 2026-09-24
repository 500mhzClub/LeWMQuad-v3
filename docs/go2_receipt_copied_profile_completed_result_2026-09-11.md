# Completed receipt-copy controller profile

The profile completed all 1,428 original observations and 1,425 forecasts.
Its owner ended; tool session 42270 exited zero. Independent completion checking
passed in session 6960, verifying 2,206 source bindings, the exact eight output
artifacts, completed predecessor evidence, actual raw/model bindings, all replay
row identities and full reconstruction of all three profiler summaries.

- Result SHA-256:
  `267446c2c7a2fea4d1b3e63887b6ee40d82d238d211eafd0c89c4b6b8f785a4b`.
- Launch SHA-256:
  `a202f29ad40951338b3016d6805131fae17ca3d5ffae5b26ec00d499896685e7`.
- Completion verification:
  `go2_receipt_copied_profile_completion_verification_2026-09-11.json`,
  SHA-256 `b94eb7781acd5a45a9ea93329e73f513a1d20fd27630f5497f9454c03d5a0119`.

| Fixed window | Frames | Total exclusive profiled seconds |
| --- | --- | ---: |
| Early navigation | 3–12 | 7.271 |
| Repeated hold | 395–404 | 11.213 |
| Late navigation | 1418–1427 | 14.778 |

The late window spent 2.642 cumulative seconds in retained floor-patch coverage,
about 17.9% of the window. Its batched projection helper ran 13,372 times; the
per-frame coverage recorder ran 422,977 times. The recorder computes visible
indices on each call before testing actual prefix-image coverage. The profile
does not count how many of those visibility rows were empty.

The next candidate is to derive one boolean visibility summary per batch and
avoid per-frame visible-index enumeration for empty rows. Preserve the original
chronological earliest-witness rule, prefix-key access even for invisible
frames, early stopping, arithmetic-error fallback and complete owned receipts.
Verify those semantics and measure the effect before adopting a controller
change. The diagnosis and exact caller counts are recorded in
`go2_receipt_copied_profile_retained_patch_diagnosis_2026-09-11.json`.

Other material costs remain: observed floor-cell index construction took 2.235
cumulative seconds and receipt freezing took 1.084 seconds in the late window.
Cumulative times overlap and must not be added. These are shared-host profiled
controller windows, not a new speedup comparison or a real-time result.

The original sensing failure at frame 1173 and failed round trip remain in the
evidence. The checker did not repeat neural inference or full training ancestry;
descriptive state-size snapshots were not independently reconstructed. This
profile establishes no new navigation, independent-layout, memory-benefit,
JEPA-benefit or hardware qualification. The full CPU replay slot is now free.

Separately, the original tracking worker began sensor collection. Its first
complete observation was checked without consuming the active stream to EOF;
see `go2_original_tracking_native_collection_started_2026-09-11.json`, SHA-256
`399483b0ebaa9d51dd11d2b1b308b9bae0a60b92febddbe7c3c09759ab5fa35b`.
That episode is still running and has no audited navigation outcome yet.
