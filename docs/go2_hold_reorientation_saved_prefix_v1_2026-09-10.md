# Fixed saved-selection boundary check

Consume only the completed first-1,000-row diagnosis, result
`1a9b9627a9496b40aaf5e20aacad00172fbc892c827bda3e4d830df5b4843884`.
Authenticate its output and inherited source hashes. Apply the separately
specified hold-reorientation state machine to successive saved selections,
resetting its hold credit for mission target changes and preserving skipped
mission/warmup calls. Stop at the first changed physical request. Preserve exact
complete original selections before that boundary; at the boundary permit only
action, index, requested command and the explicit intervention receipt to differ.

This checks the prospective decision rule on original saved forecasts and
gates. It does not reconstruct the full original or candidate controller,
execute model inference, validate raw sensor inputs, audit original completed
commands, or infer the result of executing the candidate command. No observation
after the first changed request is consumed. Full raw replay against the
completed original case remains necessary before any new native execution.

Use one bounded CPU process, eight GiB available memory, and 40 GiB reserve plus
one GiB allowance. Bind sources, input result and three output artifacts before
and after the check. Exclusive output
`go2_hold_reorientation_saved_prefix_v1_attempt_001`; preserve any failure.
The active batch and queued reached-frontier experiment continue unchanged.
