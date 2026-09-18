# Sustained reorientation native collection and audit source

The separate native collector and raw auditor use the sustained-reorientation
controller. Their entire syntax trees must equal the frozen original hold-run
collector and auditor after normalizing only the controller import/name,
experiment status strings, and the new enabled receipt/assertion. No sensor,
gait, scene, command interval, navigation budget, settling, drain, resource,
timing, visibility, contact or success condition changes are permitted.

The collector is
`scripts/sustained_hold_reorientation_maze02_episode_development.py` and the
auditor is `scripts/sustained_hold_reorientation_maze02_audit_development.py`.
The equivalence checks are
`lewm/tests/test_sustained_hold_reorientation_native_source_development.py`.
They bind both complete original files by SHA-256 and include mutations that
must fail the comparison: navigation-budget changes, missing sensor inputs,
shortened terminal drain, disabled tracker requirements, relaxed visibility
failure handling and removed raw decision replay checks.

This prepares source only. The full prospective raw replay is still a separate
prerequisite. A native launcher must also require the complete unchanged source
and model identities, the fixed observation-406 first-command boundary, available
resources, and completion of the already queued contact/tracking/extended-budget
experiments. Do not launch this collector directly or overlap native scenes.
No replacement of the independent-study policy is selected by this source work.
