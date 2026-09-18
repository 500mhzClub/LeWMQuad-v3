# Copy-cost probe on saved footprint receipts

The completed frozen-footprint profile places its largest remaining external
`deepcopy` costs inside `LaterResolvedFloorMemory.footprint` and
`confirmed_contact_check`. The later window attributes 2.033268511 and
1.429569512 cumulative profiled seconds respectively to their copying calls.
Profiler overhead is included; these are not unprofiled controller timings.

Probe the existing `lewm/receipt_copy_development.py::copy_receipt` against
standard `copy.deepcopy` on saved subreceipts at original JEPA observations 3
and 404. Each group contains the six action-specific receipts. The later-floor
group is `original_contact_check_before_later_floor_resolution`; its nested
`original_auxiliary_floor_contact_check` supplies the confirmed-auxiliary group.

| Original observation | Receipt group | Standard copy median | Existing helper median |
| --- | --- | --- | --- |
| 3 | Later-floor original, six receipts | 10.78455 ms | 5.67220 ms |
| 3 | Confirmed-auxiliary original, six receipts | 5.30115 ms | 2.83062 ms |
| 404 | Later-floor original, six receipts | 14.32274 ms | 7.60459 ms |
| 404 | Confirmed-auxiliary original, six receipts | 6.52788 ms | 3.68927 ms |

Each workload used two warmup repetitions and six measured repetitions, with
alternating implementation order. Serialization and equality checks were outside
the timed region. All complete serialized copies matched, and each copied root
was independent of its source. These samples come from serialized receipts;
the existing helper's separate tests cover aliases, cycles and custom-copy
semantics that serialization cannot preserve as identity evidence.

Session 88219 exited successfully. The completed original decision stream was
hashed before and after reading observations 0–404, retaining only the two
selected observations. Its SHA-256 is
`8c4fbbfa17cefb1e29399a07bb735b919a2f8554fd4b30a87e6409426b090ca4`
at `all_phase_full_jepa_residual_maze_02/context_decisions.jsonl.gz` in the
existing adapter six-case development root. All 2,020 current verification
source bindings were rechecked. No sensor packet, model, controller or native
scene was executed in this probe.

The helper reduces these individual copying workloads by approximately 43–47%,
but saves only about 2.5–6.7 ms per six-receipt group. This is a possible focused
optimization inside footprint construction, not evidence of a large full-loop
gain. It does not justify claiming real-time behavior or changing the running
replay. Wait for the current combined mesh-reuse result before selecting another
complete-controller experiment.
