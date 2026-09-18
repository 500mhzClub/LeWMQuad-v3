# Expanded input-stream preparation and running validation

The preceding goal turn completed4010 available expanded target contexts.
This turn implemented a separate training-only input stream and started its
complete real-data validation. The original native queue remains unchanged.

New source bindings:
- scripts/all_phase_training_policy_stream_development.py:302a230c16196cf685a5ffafe40cc06402388c8b05cb017e736f4b1b25a7a5ef
- lewm/tests/test_all_phase_training_policy_stream_development.py:af67cdfde828d82437aba8dbfbbc6239475c3b0961a95197b7eadb87e37db981
- scripts/check_go2_all_phase_training_inputs_v1.py:74259e5db970b165b046e2b690f426ce71f2363a1f3afb3c1375efedea5d72b6
- docs/go2_all_phase_training_inputs_v1_2026-09-10.md:522f53ab8082b0c205b84ddedabe4d750e5b0f91c155d3b86b9e7405682cc922

The stream preserves original tensor normalization and action masks while
admitting offsets0..39. Its inference path reads only four declared past
packets and does not inspect target labels. A distinct scoped reader loads
private training-only futures. Every consumed policy leaf is bound and checked
before/after materialization. Native target labels remain outside inputs.
The optional LRU tensor cache has an8GiB ceiling, actual byte accounting,
eviction and fresh stacked batches; the full checker uses no sample cache.

Initial43 focused tests passed; adding exact derivation-receipt checks brought
the final suite to45 passing tests in2.46s, handle96108 closed0. Coverage includes
all offsets and shared old action plans, separate actual reader scopes,
past/future clocks and identities, inference with inaccessible target labels,
role leakage, corrupted bindings, target masks, failure latching, cache eviction
and mutation isolation. The original1803-source queue and1825-source contact
waiter unions were checked unchanged after the new files were added.

Preparatory real-data check75811 successfully constructed six actual samples
in0.4601188278757036s but stopped in its untrained model-interface check because
the human label supervised was not the repository condition supervised_rollout.
It created no owned output and performed no optimizer or parameter update.
The checker now imports the original CONDITIONS tuple. Corrected preparatory
check87818 closed0: six samples in0.44758349005132914s; direct,
supervised_rollout and JEPA all have finite active gradients and unchanged
matched untrained initialization
ed2c1f096b430b424cf6e381047eeee7672934bcac2fdbe198d5f85c2cead607.
No optimizer was created and no model was fitted by these checks.

The actual full-population check is handle80757,
PID2628968/starttick133510189/create_time1789014068.89.
Output root go2_all_phase_training_inputs_v1_attempt_001.
Launch SHA b8eb7241b54842dd10d9a55d97752a6ad8f4e4514e8bcf5e32b4551e3da1e090,
1123 sources. It has reported2688 materialized contexts and remains live.
Latest observed RSS1,356,824,576 bytes and CPU time298.67s show active progress.
Do not restart it or edit its frozen sources. Full outcome remains pending.

Metadata discrepancy found while it runs: the broad inherited launch includes
future_rgb_materialization=false and future_rgb_materialized=false from the
parent target derivation. Those two flags incorrectly describe the current
checker. Its new explicit private_training_future_materialization=true,
geometry_transfer_future_materialization=false, prospective protocol and
separate scoped readers define the actual authorized operation. Preserve the
original launch and outcome, then verify every access receipt and issue a
hash-bound scope correction before admitting this result for fitting. Do not
silently overwrite either flag or claim the uncorrected metadata is consistent.
Future launch builders should copy environment identities explicitly rather
than inherit unrelated stage metadata.

The existing loss already scales XY by0.06m. It is not an unscaled metre loss.
The original matched trainer uses AdamW at fixed0.001 learning rate, zero weight
decay, gradient clipping1, EMA0.99 and1200 updates. Its data/view/fit helpers
hard-code912 old slots,408 train contexts and five-tick offsets; they cannot
be silently repurposed for this expanded data. Subsequent work needs a combined
view with original transfer-role identities, new plan validation and explicit
matched schedules before fitting. A full4010-sample cache is expected to need
about7.36GB of tensor storage; use the completed check's measured byte report
and fresh hardware/throughput assessment to choose fitting concurrency.

Full checker80757 subsequently exited0 with result
ef48950b7987eaf9310ba8124a00c2e6e13c9b84b7cda6a362ac7ddb6ecd63fb.
Every4010 available input and all408 original-input witnesses passed. The
independently verified metadata scope correction is
docs/go2_all_phase_training_inputs_scope_correction_2026-09-10.json,
SHA8fcfdc678b721a84ed53c2545ec55284e2185d7b1e6bc5ca0dc26da4832e2dca.
See docs/go2_all_phase_training_inputs_result_2026-09-10.md for complete output
bindings, verification scope, cache measurements and the next fitting work.
Do not restart the closed checker or alter its original launch flags.
