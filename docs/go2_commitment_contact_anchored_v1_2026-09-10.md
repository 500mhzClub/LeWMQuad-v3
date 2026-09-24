# Commitment contact cost in ordinary anchored waypoint selection

The expanded-data supervised maze-02 collection exhausted its mission budget
with no forward command or edge crossing. Its full raw audit remains pending.
Saved early decisions reveal positive predicted first-interval forward progress
but a larger full-800-ms contact penalty. At observation 14, predicted executed
forward distance progress is 0.014259773206678261 m while the 800-ms contact
score is 0.04676586101514618, multiplied by the unchanged 1.2-m coefficient.
Thus the contact term exceeds the available first-interval pose benefit.
Those six actions pass the recorded phase and geometry checks. This is an
observed scoring tradeoff, not evidence that future motion is physically safe.

The existing `score_commitment_contact` component already scores pose and
contact over the actual 100-ms command. It preserves coefficient 1.2, full
forecasts, all eight 800-ms nominal path gates and articulated surface vetoes.
Its earlier old-model supervised maze-01 trial, result
`90116b248154b9d4731501d5c6cc6a317345101b9f39c6ed5ce712c9baa32986`,
produced translating commands but failed on current visual evidence at
observation 140, without a goal arrival or maze-edge crossing. That negative
result must remain visible. Do not present this component as new or successful.

Reuse its exact checked source in `CommitmentContactAnchoredController`.
Run the full original anchored selector first. If any first-interval, hold or
anchored residual recovery receipt is active, return that original selection
unchanged. Otherwise apply the existing commitment-contact scorer, including
its exact reconstruction of the inherited executed-waypoint selection.
Existing view acquisition, final-goal and nominal-clearance reentry scopes pass
through unchanged. The integration therefore changes the contact cost only for
ordinary intermediate-waypoint choices; it is not a uniform contact-horizon
change inside the residual recovery calculations.

All original map, motion estimation, memory, residual, mission, model and
observation implementations remain inherited. It does not include batched floor
queries, receipt-copy optimization, hold reorientation or frontier retirement.
The controller declares its identity and
`ordinary_waypoint_commitment_contact_enabled=True`. Scores remain uncalibrated.
Late-horizon learned contact cost is retained as evidence but is not charged in
these ordinary choices; this is a substantive risk/utility-policy intervention,
not an implementation-only speed optimization. The unchanged geometry gates
do not certify physical safety or future replanning availability.

Prospective bounded preparation uses only the already assigned full-supervised
expanded-model maze-02 case and model
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
Do not select a model, layout, coefficient or horizon by sweeping saved outcomes.
A saved-prefix boundary check must stop at the first changed command and compare
complete candidate selection against exactly `ordinary_commitment_contact`.
It is not a fresh neural replay or an executed candidate trajectory. Recorded
original observations after that boundary are not candidate observations.

The saved checker is
`scripts/check_go2_commitment_contact_anchored_saved_prefix_v1.py`, with exclusive
output `go2_commitment_contact_anchored_saved_prefix_v1_attempt_001`. Bind the
original batch launch, fixed model assignment, provisional collection readout,
complete saved decision stream and completed command tape before use and again
after checking. Its source-only preflight creates no output. This lightweight
saved check requires 8 GiB available RAM and 41 GiB artifact headroom; it does
not perform the original full transitive native-input admission or claim a
completed original raw audit. Preserve any failed attempt without restarting.
The tests include a generator that raises if a single original observation
after the first changed command is consumed.

Before any prospective native use, require completion of the original supervised
raw audit, exact input/model/source admission, and a fresh paired raw replay
with separate models/controllers. Compare full public observations, model
forecasts, geometric vetoes, observed mission/map/residual state and original
commands through the first intervention. Pending forecast for the newly selected
command must refer to that action; no unobserved successor label is compared.
`commitment_contact_anchored_prefix_development.PrefixComparison` enforces full
decision equality apart from the declared selection and implementation metadata,
and latches at the first different command or terminal.

Any future native trial needs a separately named prospective launcher and full
physical/public prefix comparison, followed by the original raw reconstruction,
contact, visibility, goal/return and timing audit. No native scene is launched by
this preparation. Existing six-case, frontier and hold-reorientation execution
and ordering remain unchanged. This preparation makes no real-time,
independent-layout, hardware, safe-exploration or navigation-success claim.
