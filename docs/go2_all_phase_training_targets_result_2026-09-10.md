# Expanded training target result

Completed result:
4d300f77849d174cc9d7bd2a35d276e0996d1b5ac8795bee4f419291ada6b328.
Root: go2_all_phase_training_targets_v1_attempt_001 under the owned navigation
development artifact root. Status: ALL_PHASE_TRAINING_TARGETS_V1_COMPLETE.

The original120 training trials (48 family,72 switch) supply4010 available
contexts from4800 fixed offset slots, versus408 available contexts used by the
existing fits. All456 original training slots retain their exact availability
and targets. There are3974 first-step motion labels and3140 complete eight-step
motion sequences. Missing contexts remain in the4800-slot denominator.
There are1443 valid first-step motion labels with a zero next command.

| Control phase (camera index modulo5) | Available family | Available switch | Total |
| --- | ---: | ---: | ---: |
| 0 | 334 | 471 | 805 |
| 1 | 330 | 464 | 794 |
| 2 | 328 | 460 | 788 |
| 3 | 336 | 476 | 812 |
| 4 | 336 | 475 | 811 |

Every recorded command transition still occurs in phase3:84 family and115
switch-trial transitions, including28+49=77 nonzero-to-zero transitions.
Expanding context indices adds neither command-switch-phase diversity nor
independent episodes. No geometry-transfer/nav labels were added to training.
No model was trained, future RGB loaded, native scene collected or inference
performed by this derivation. The output is target metadata, not verified model
input tensors or a trained navigation system.

Output identities:
- launch.json:fa468fe4b22eaa83a859ca5a21dac9f4e0e2c2b0093d70c2201718bbf5728ad1
- windows.json:fcf0ae850604259195bf36a812e9f47caefb63936bff9c2476d3cd52315e901e
- coverage.json:c7d56c50353e8537721bcc5a84fda8d9911bd0854d8e814c87e3dd48e9f2e6d1

The training-only exploratory census27945 exited0 with the same denominators,
all456 original targets exact and1347 original-source/consumed-file bindings
checked before/after. The new derivation and original target tests then passed
14 tests in1.88s (81884 exited0), covering all phases, complete known-command
suffixes, causal history indices, old-label preservation, retained post-contact
contexts, role mixing, duplicate/missing original slots and changed physics
commands. Hardware before execution showed77,503,389,696 available RAM bytes,
3.4% CPU utilization, zero GPU utilization and671,756,644,352 artifact free
bytes. One CPU derivation ran beside the original single native worker.

Initial invocation48685 failed before creating any output because the new
launch metadata omitted the original verifier's input_sha256 fields. Output
absence was explicitly verified. The preparatory fix retains the original
launch metadata and replaces the declared derivation fields. Actual exclusive
derivation33328 then exited0 in13.099572757957503s, with1108 bound sources and
240 consumed training physics/camera leaves. Total output size including result
is38,847,557 bytes, within64MiB. No failed native or output attempt was replaced.

Independent completion verification checked all1108 sources before/after,
all three outputs, all240 consumed training leaves, exact launch/source and
coverage/result agreement,4800 distinct sample IDs and exclusively unchanged
training roles. Historical raw collection/controller audits were not rerun;
the explicit scope is bound source/artifact identity and exact native target
derivation from the previously audited recordings.

Next: implement and validate a separate policy-only input stream for all new
offsets. The old plan() permits only five-tick offsets and the old view/cache
assumes408 training contexts, so those frozen classes must not be reused as if
they already support the expanded data. Retain their tensor normalization,
four-observation causal histories and eight100ms action blocks. Compare all408
old available input tensors exactly, validate new past/future clock and role
boundaries, and bound private training cache memory. Keep original
geometry-transfer evaluation assignments. Review training loss scaling and
sample schedules before freezing matched JEPA/supervised retraining; do not
start training merely because target derivation succeeded.
