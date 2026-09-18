# Independent population entry point with eight-diagnostic review

`scripts/run_go2_independent_round_trip_population_eight_diagnostics_v1.py`
provides a separate launch path for the original fixed 32-case population if
the eventual eight-diagnostic review explicitly retains that definition. It
does not change the four arms, eight layouts, case order, assigned models,
3,000-tick budget, collectors, auditors or monitored runtime.

Source preflight authenticates source/launch bindings and current resources;
it reads no diagnostic outcome or final review and creates no population output.
The required resource envelope remains the original 392 GiB for all 32 cases,
at least 64 GiB available RAM for the staged driver, and one native scene.
These checks are measurements, not resource reservations.

The input-bundle preparation helper first rejects all live diagnostic owners.
It authenticates the original batch and five diagnostics through the original
bundle preparer, then appends all three later diagnostic outcomes through the
new evidence layer. It returns an explicitly unapproved bundle for a separate
scientific review. The expected bundle path is
`docs/go2_independent_round_trip_eight_diagnostic_completed_input_bundle_v1.json`.

Actual preflight or execution requires exact SHA-256 bindings for that bundle
and the new eight-diagnostic final policy review. Both documents join the source
closure. The launcher requires full final admission and independent overlap
admission before creating the exclusive output
`go2_independent_round_trip_population_eight_diagnostics_v1_attempt_001`.
It passes the actual new final verifier to the original monitored driver.
Successful flags alone cannot bypass either verifier. Terminal failures and
existing original failure artifacts are preserved without retry or resume.

Preparation is not a launch decision or navigation result. Review actual native
and timing outcomes before choosing the population definition. If a revision
is needed, implement and check it prospectively before consuming independent
sensor data. The original comparison's method-level reactive and planning-grid
memory scope remains unchanged; no isolated prediction-ranking or fully
memoryless claim is introduced.
