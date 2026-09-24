# Completed no-RGB JEPA maze-2 case: five edges, visual tracking failure

The first no-RGB JEPA case completed collection and its original full raw audit.
Independent authentication verified the exact worker and parent completion,
1,908 original source bindings and 5,260 artifacts, reconstructed paired startup,
recomputed the complete native evaluation from the physics trace, and reproduced
the saved readout. The development ledger now contains 41 completed audited
episodes and zero verified round trips. Four of the six fixed comparison cases
are complete; the no-RGB supervised and direct cases remain.

This case traversed five distinct declared open edges, with no invalid edge
crossings or native contacts. It reached no observed or native goal arrival and
performed no return traversal. The first terminal decision is frame 859:
`SENSOR_OR_MODEL_FAILURE`, with controller failure
`same-episode current visual evidence required`. Ten subsequent zero-command
intervals completed. The record contains 870 observations, 869 command intervals
and 44,200 physics samples.

There were 856 model-selected navigation actions: 294 left arcs, 24 right arcs,
330 right turns, 92 holds, 61 left turns and 55 forward commands. At the last
admitted mission frame, observed distance to the outbound goal was about
3.2042 m. This establishes useful movement on this reused development layout;
it does not establish successful navigation or a model advantage.

Strict physical visibility, raw sensor reconstruction, raw model-command replay
and command audit all pass. There are no hard-measurement failure frames. The
assigned model state remains
`fb6f1aba8830a53d67cd6c284fb24199966d5f0c63db3b2a107ab833c81c266f`.
The no-RGB treatment applies to the learned model; the controller retains RGB-D
and the auxiliary camera for visual localization and mapping.

Recorded visual evidence identifies the immediate failure before floor
registration: frame 859 has no accepted retained-anchor or previous-frame pose.
The incremental failure is `insufficient rigid consensus after pruning`.
Frames 853–858 had continued on measured incremental bridges, reaching six
bridge frames. The next diagnosis should inspect actual primary/auxiliary
correspondences and the original pruning/consensus checks at this transition.
The generic controller error alone does not identify a model failure. No gate
was relaxed, failed pose accepted, or original case restarted.

Across all 870 observations, acquisition plus controller median time was
1024.1259515 ms, p95 1776.63094985 ms and maximum 2418.744525 ms. All exceeded
the 100 ms command interval. Median iteration time including receipt persistence
was 1077.5982965 ms. Original audit position-error samples (859 valid observations)
had median 6.811877686 mm and maximum 12.412671381 mm. These are simulation
measurements with physics paused during computation, not hardware or real-time
qualification.

Independent authentication did not rerun neural inference, visual fitting, the
full original raw audit, or training-input admission. It authenticated those
original receipts, read the complete decision stream, and independently
reconstructed native traversal/contact evaluation from recorded physics.

Artifact root: `go2_all_phase_adapter_maze02_matched_native_v1_attempt_001`.
Case: `all_phase_no_rgb_jepa_residual_maze_02`.

- Worker terminal: `5bc0c40bda4435bfaf6e6616e81981d23b1d6616f0530435b1a5c74dba4dad60`.
- Audit: `fd1eb305f90eab203651878a1c88eb72e3993c74c19d9a830b3bcc7d3ae22c5b`.
- Readout: `860f9f6df9466f7adc16c9bc4dc99063064175b3ad3f4a6738023f9d6369fe6b`.
- Parent completion: `d3597bc5834afbafbf9e9bb56b8c35fa92346d2ddd98cbb2a21fc04d2224d93a`.
- Independent verification: `f2e2cdb317d913eecaee7b49712e3133ed78306ab10a64e99b14f1721d148a18`,
  `go2_all_phase_adapter_no_rgb_jepa_maze02_verification_2026-09-10.json`.
- Recorded pre-failure visual context:
  `go2_all_phase_adapter_no_rgb_jepa_maze02_visual_failure_2026-09-10.json`.

The original worker ended. The parent started its next assigned worker,
PID 2743870, creation epoch 1789071424.56. Frontier, hold-reorientation and
contact-score experiments remain queued behind the complete six-case batch.
The goal remains active and incomplete.
