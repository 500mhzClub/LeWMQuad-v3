# Paired replay implemented; resource preflight did not pass

The next paired replay is implemented in
`scripts/replay_go2_packed_fused_scoped_late_history_v1.py`. It will compare the
fused receipt controller with the packed-index composition over all 1,428 old
observations, with 1,425 forecasts and seven exact retained-state checkpoints.
Its completed-predecessor checker also provides a concrete reader for the
currently running fused replay when that original process ends.

Fifty-six tests passed in 9.07 seconds, session 75052, exit 0. Coverage includes
the full synthetic paired history, changed reference/input/model/state
rejection, exact original process identity, no runtime during source preflight,
and preservation of failure after final input changes. Source discovery and
identity checking cover 2,183 paths. The actual live predecessor is correctly
rejected by the execution admission gate.

Source preflight session 67945 exited 1 at the hardware resource check.
Available RAM measured immediately afterwards was 61.4157 GiB, below the
64 GiB requirement. Artifact storage had 576.5 GiB free and the host had 16
physical CPUs. The requirement remains unchanged. Neither the new experiment
root nor any new replay/native execution was created. Recheck resources after
the existing fused replay releases its slot; an observation timeout is not
grounds for restarting that process.

The first preparation record incorrectly labeled the failed preflight as
exit 0 because it was written before the failed tool result was inspected.
Its original bytes are retained under the explicit name
`docs/go2_packed_fused_scoped_late_history_preparation_misrecorded_preflight_2026-09-11.json`,
SHA-256 `3ebe72f5c27ccd8274f5a04477ca9ec0e27957bba6fc69cbb2318ce30a961a1f`.
That record is invalid as evidence of a passing preflight.

The corrected authoritative record is
`docs/go2_packed_fused_scoped_late_history_preparation_2026-09-11.json`,
SHA-256 `5c4f954b2c56568953df7f7a39fe90f5c631bdb462648cd10feff7aca67f9d11`.
It records the actual failed preflight, the valid source/test results and the
preserved incorrect record. No current experimental source or output was
changed. This preparation establishes no new speedup or navigation result.
