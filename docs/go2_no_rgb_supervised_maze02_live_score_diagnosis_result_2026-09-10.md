# Live no-RGB supervised case: score-driven lack of translation

The fixed completed decision prefix through observation 2635 contains 2,636
rows: three warmup rows, 2,563 holds, 60 left turns and ten right turns. No
translating action was selected. All 2,633 planning rows report six admissible
candidates. The longest uninterrupted hold sequence is 240 observations,
frames 39–278. The observed mission reports no arrival and remains outbound;
goal distance changes from 4.687217 m to 4.687455 m. These are saved online
observations, not a final independent physical readout.

The diagnostic authenticated the original batch launch and 1,908 source
bindings, checked ordered complete rows and requested commands, and recorded a
canonical hash of the fixed row prefix. It reconstructed all six candidate
utilities at frames 3, 2633, 2634 and 2635 from saved forecasts and residuals:
24 exact score comparisons. All six candidates have clear recorded surface and
nominal-path checks at those four sampled frames.

At frame 2635, forward's potential progress is 15.093 mm, but the 800-ms
contact score incurs a 32.938-mm penalty, leaving utility −17.845 mm. Hold's
utility is −3.535 mm and therefore wins. The predicted contact scores are not
calibrated probabilities. This reproduces the score calculation; it does not
establish physical safety or that forward would succeed.

[Fixed-prefix score diagnosis](go2_no_rgb_supervised_maze02_live_prefix_score_diagnosis_2026-09-10.json)
has SHA-256
`6cb4683d9cf77c1d24eb00da7a35a53ddba48feccb2b2e28922115b6a57b6509`.

A separate probe consumed only observations 0–3 and invoked the already queued
contact-horizon score helper on the first saved selection. The helper reproduced
the complete original selection, then changed `left_turn` to `forward` when
using 100-ms contact cost. Forecasts, surface checks and all eight nominal path
checks remained unchanged. Forward utility changes from −2.686 mm to 29.846 mm.
This is a saved-selection calculation, not a full-controller replay, neural
inference, changed-command execution or counterfactual trajectory. No following
recorded observation was consumed by that probe.

[First-selection probe](go2_no_rgb_supervised_maze02_first_contact_score_probe_2026-09-10.json)
has SHA-256
`8c33aa4afb2ebb3983cec6c524f9a05a0560359f349110377c07e5d2f3b5901c`.

The evidence supports prioritizing the existing prospective contact-horizon
test rather than adding another feasibility relaxation for this particular
stall. Its scheduled native case remains the full-sensor supervised model;
this diagnostic does not expand that case or demonstrate transfer to no-RGB
execution. Preserve the running fixed batch and wait for its complete raw and
physical audit before counting the fifth case as a completed episode.
