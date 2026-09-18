# Receipt-copy controller benchmark complete

Session78762 exited0. Result SHA-256
60849db8aa7a2b09e5e6de977115cb953f6c1b893ed62b0533ccd19b31f1ff95;
launch96f6a1b90454689d0fe8b6f602ba178535be58054d2401e45553f586e8ea569e;
paired streamf7cb235ba3a7e15e4fba1ee1f4ec381abaabfa9d149ae25f6f99ffbd96365b00;
resource monitor113b8e8e5e75d1ca3e2c3b8421f8d870dfaea4790d976d7b30204936268e49ef.
Root:go2_receipt_copied_controller_benchmark_v1_attempt_001. All1,671source
bindings and original input/artifact checks completed; result and output artifact
hashes were checked again when reading these findings. Wall time983.4213553s.

All514 complete original and candidate decisions exactly reproduced the saved
original maze2 trajectory. Both model states and public inputs remained unchanged.
The copied selector preserves the original code, closure, defaults and state flow;
the 15 explicitly changed dependency bindings are recorded in its preparation.

For500active observations, excluding3warmup and11terminal observations:

| Measurement | Original | Receipt-copied |
| --- | ---: | ---: |
| Median controller time | 750.178271ms | 695.7320085ms |
| Mean controller time | 776.9140114ms | 723.6661423ms |
| Observations exceeding100ms | 500 | 500 |

Median paired reduction54.313986ms. When original ran first, the250-observation
subgroup's median paired reduction was57.693716ms; when candidate ran first,
51.321832ms. Both order subgroups show a reduction. These are sequential paired
measurements on one fixed completed episode, with concurrent native work. Packet
reconstruction, sensor acquisition, hashing, comparisons and receipt I/O were
outside the timer. No profiling instrumentation was used.

This establishes complete-decision equivalence and a modest controller-replay
improvement on this episode. It does not establish whole-loop speedup or real-time
operation: every active candidate cycle still exceeds100ms. No native controller
has been changed or promoted, and no new navigation outcome was generated. The
remaining large performance costs are outside these selected receipt copies.
