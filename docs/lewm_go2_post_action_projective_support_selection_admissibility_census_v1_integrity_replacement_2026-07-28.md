# Selection admissibility census V1 integrity replacement

Date: 2026-07-28

The invocation authorized by
`lewm_go2_post_action_projective_support_selection_admissibility_census_v1_authorization_2026-07-28.md`
stopped while parsing the exact V4 execution binding. It opened only that binding
and raised `LabelContractError: exact V4 execution binding content changed`. It did
not open raw metadata, geometry, source frames, RGB, labels, models, checkpoints,
GPU/runtime outputs, navigation, G2, held-out, or sealed material and wrote no
output.

The bound file's exact verified bytes end in one newline. The diagnostic compared
those bytes to canonical JSON without its terminal newline. Source commit
`7e79346253c707cba440cf1557649d3d6adca844` changes only that comparison to require
canonical JSON plus exactly one newline. The corrected diagnostic is 14,324 bytes
with SHA-256
`ed6f7c8eec6b23d387bcc615859c7b0c7fccb351beabce7d9e23f96567bdad94`;
its focused test remains `1 passed`.

Authorize exactly one integrity-replacement invocation of the same command. All
scientific definitions, inputs, roles, decision thresholds, access prohibitions,
stdout-only output, and downstream authority remain unchanged. No further retry
or resume is authorized by this replacement.
