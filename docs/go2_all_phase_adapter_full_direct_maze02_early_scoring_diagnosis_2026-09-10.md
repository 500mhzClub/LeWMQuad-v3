# Direct-model early scoring diagnosis, audit pending

The direct model shows the contact-scoring imbalance already diagnosed for the
supervised model. This finding comes from two saved decisions in the first 15
observations. It is not a completed episode audit or a new policy trajectory.

At observation 14 all six actions pass the recorded phase, surface and nominal
path checks, yet holding wins the original score. Forward motion predicts
20.094 mm of progress over its 100 ms commitment. Its original utility is
-29.837 mm because the cost uses the 800 ms contact score. The existing
commitment-contact scorer instead gives forward motion +18.703 mm, while holding
scores -0.098 mm. The unchanged contact coefficient is 1.2 m; these contact
scores are uncalibrated and are not collision probabilities.

| Original observation | Original action | Action after saved-selection rescoring | Original forward utility | Rescored forward utility |
| --- | --- | --- | --- | --- |
| 3 | Left arc | Forward | -19.498 mm | +16.710 mm |
| 14 | Hold | Forward | -29.837 mm | +18.703 mm |

The existing scorer reconstructs the complete original executed-waypoint
selection before rescoring. Both inspected selections preserve every raw
forecast, surface check, nominal path check, allowed action and causal residual
receipt. This inspection does not rerun model inference or sensor processing.
Observation 14 is conditioned on the original logged history: changing the
command at observation 3 would require new physical observations. No outcome
for that alternative trajectory is inferred.

Evidence: `docs/go2_all_phase_adapter_full_direct_maze02_early_scoring_diagnosis_2026-09-10.json`,
SHA-256 `76e8d96811d374704ca5617bc04a5678a35e211213d67a21248e2f5af6945efa`.
The first 15 canonical rows, exact collection result and complete compressed
decision stream are bound in that record. All 2,017 recorded source bindings
and inspected artifact hashes were checked before and after the diagnosis.
The direct case and model assignment match the frozen native launch and adapter
admission. The model itself was not reloaded.

Collection ended after 3,000 navigation ticks, with 3,014 paired observations,
3,013 completed command intervals and 151,400 physics samples. It reports
mission-budget exhaustion, ten zero-command drain ticks, and no physical or
acquisition stop. The worker's raw audit remains pending; these collection
facts do not establish valid sensing, navigation success or a completed case.

This supports continuing the already queued commitment-contact policy test on
the supervised model. It does not add another native attempt or change the
six-case batch, frontier, hold-reorientation and commitment-contact ordering.
