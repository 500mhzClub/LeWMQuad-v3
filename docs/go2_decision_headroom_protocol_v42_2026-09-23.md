# V4.2 — approved output conversion and GPU allowance

The [V4.2 configuration](go2_decision_headroom_protocol_v42_2026-09-23.json) has SHA-256 `78b9b7d0ff79d7328cfc0511126e440a9bc467e9fdee08cdaa3378cc2c95acba`. Its [approval record](go2_decision_headroom_v42_approval_2026-09-23.json) contains the user's exact text. The [diff check](go2_decision_headroom_v42_diff_check_2026-09-23.json) confirms that differences from V4.1 (`e7735ae9…`) are confined to version metadata, the output converter and implementation bindings, and the GPU-owner allowance. All previous bindings and other caps remain unchanged.

The single converter uses `.item()` for NumPy booleans, integers and floats, and `.tolist()` for NumPy arrays. Other values retain their existing types and values. Ordinary JSON semantics remain explicit: Python tuples become JSON arrays, and legal scalar dictionary keys use JSON's key spelling. No scientific values are rounded, zeroed, imputed or recomputed by conversion.

The audit owner installs this boundary for inherited JSON and JSONL writers, including spawned workers. Serialization is compared against the in-memory structure and values. Actual written bytes are immediately read back; completed JSON documents and JSONL records are parsed and validated. Required audit envelopes have explicit field and shape checks. Any mismatch or writer failure raises a stop that cannot be swallowed by a state-local `except Exception` handler.

The GPU-owner allowance is now **72 hours**, matching the existing 72-hour execution wall cap. CPU remains **1,152 core-hours**. Collection, branch, RAM, VRAM, storage and filesystem-reserve limits are unchanged.

The implementation check invokes the complete V4.2 owner with `--implementation-check`, in `go2_headroom_v42_output_check_attempt_001`. It supplies the six hash-bound qualified packets and retained branch traces to the ordinary branch owner and frozen models. It then invokes the ordinary reader and assembles a versioned branch panel and memo inputs from the written files. These records are implementation-only and must never be merged into the Phase 2 panel. No new physics is needed for retained assignments.

The separate historical re-render owner follows this check, under its unchanged historical caps. Missing historical restoration inputs remain unresolved and do not gate Phase 2. Any different implementation defect stops for disposition. No controller changes, model fitting, additional states or artifact retirement are authorized.

Execution status and measured validation evidence are recorded separately so this protocol identity stays frozen.

The [complete-owner output check passed](go2_decision_headroom_v42_output_check_result_2026-09-23.json): six states, 3,189 validated files, 164.66 s wall, 248.79 s CPU, 7.53 GiB peak aggregate RAM, 3.77 GiB sampled total VRAM, and zero new physics. Both available and unresolved reference paths were exercised. The historical check inspected 16 examples/32 source-image identities but rendered zero frames because qualified restoration inputs were unavailable; training-render provenance remains unverified.

[Phase 2 execution status](go2_decision_headroom_v42_execution_status_2026-09-23.json) records the existing owner and output root. Phase 2 has launched. Monitor that owner; do not start a duplicate or restart a failed assignment.
