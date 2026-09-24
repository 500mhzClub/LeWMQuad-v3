# Checker-only correction: native stop exactly at a command boundary

The physical panel completed144/144 without a collection-integrity failure.
The first full auditor then failed on trial50 after49 verified trials, with
`continuous live history coverage`. Preserve its FAIL artifact:
`.generated/go2_successive_choice_maze_development_v1_attempt_001/raw_artifact_audit.json`,
SHA-256 `19c67f2b3fea7f614cacafcb1d480b021349eca655446b4de8f37952cb862fbd`.
Its source, wrapper, tests, prior interim36 PASS and all physical data remain
unchanged. No new physics, model fitting or policy choices are authorized by
this correction.

Trial `successive-choice-maze-development-v1-02-right-direct_direct` makes the
boundary case explicit. Native contact occurs at physical sample4099, time8.2 s,
exactly50 samples after the command's pre-index4049/time8.1 s. The recorder stores
the sample, then `AttributedSession._sample` raises `PhysicalStop`. Therefore
`command_tick` does not return and its caller cannot ingest the post-command
packet. The last ingested camera index is66/time8.1 s. A terminal evaluation image
is still retained at index67/time8.2 s. No subsequent policy decision or physical
step is made. The original auditor incorrectly equated50 recorded samples with
successful return to the post-command ingestion callback.

The V2 checker changes only this expectation: omit a full-tick post-command
ingestion at the final physical sample when the recorded terminal reason is
native contact or body-stability stop. Keep that image in camera/sensor/raw-data
auditing. All completed nonterminal control boundaries must still be ingested;
a contact during release cannot erase earlier valid control history. Synthetic
tests cover full nonterminal, exact-clock terminal, early terminal, release-phase
contact and genuinely missing nonterminal images.

Write a new `raw_artifact_audit_clock_boundary_v2.json` and a matching source
dependency witness. Bind the failed predecessor audit and unchanged helper
identities. Re-audit the same144 recorded trials; do not replace or rerun them.
No threshold, action, outcome label, group comparison or physical result changes.
