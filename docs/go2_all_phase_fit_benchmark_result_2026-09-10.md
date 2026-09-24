# Expanded-data full-cache fitting benchmark result

Result2e74b02b76038f92a8b74256c083cf8ecfbe28d5a5c1bc07c5cbafda214b054f,
root go2_all_phase_fit_benchmark_v1_attempt_001,
ALL_PHASE_FIT_BENCHMARK_COMPLETE. Original15290/PID2633175 closed0. Launch
961f7fadf5f955f3fecc26c1a9494e71a31abe81575a1c330ebaf4762a53be8e,
1144 sources/47 outputs. Independent verification43508 closed0 after before/
after source/output checks and exact reproduction of the execution decision.

All six serial/parallel benchmark pairs have identical complete update ledgers
and final fit/model records. Each phase performed120 optimizer updates across
six20-update fits,240 total. No scientific model/checkpoint was retained for
native use. Each of six worker assignments materialized all4010 training
contexts into a private7,359,601,120-byte tensor cache and reused it for its two
fresh models. Source, input and correction identities remained fixed.

Measured full-phase durations:serial886.0113726989366s,
parallel337.4036113829352s; speedup2.6259688480137835. The preregistered decision
selects three full-fit workers. This is full benchmark-phase throughput,
including cache warming and worker admission/accounting; it is not a kernel
speed measurement or a guaranteed speedup for the full18-fit workload.

Worker cache-warm durations, seconds:
serial212.4372640750371/222.53617318277247/212.71765629504807;
parallel250.39500427781604/252.29182858602144/252.29060677881353.
Reported worker peak RSS, bytes:
serial9,176,956,928/9,149,771,776/9,179,586,560;
parallel5,460,975,616/5,545,132,032/5,548,064,768.
Every measured peak is within the10GiB worker allowance. The tensor-cache byte
count is a logical tensor-size count, not proof that every byte stayed resident
in physical RAM. A later host check showed5.1GiB swap used; no baseline permits
attributing that usage to this benchmark. No OS memory/swap setting was changed.

Resource monitor:57 serial samples,minimum available RAM70,822,662,144 bytes,
maximum sampled CPU busy10%;21 parallel samples,minimum available RAM
64,585,486,336 bytes,maximum sampled CPU busy16%. GPU fitting was not used under
the fixed CPU trainer/checkpoint contract. Native and verification work remained
live during portions of the measurements and is recorded in the monitor.

One-owner waiter6946/PID2635725 was registered while the original benchmark
process was live. Waiter launch4494e8370968c2643eec51f417d05be658bda747286f3a8b072bd67d9e6e3acd,
1146 sources,root go2_all_phase_matched_fits_wait_v1_attempt_001. It binds original
benchmark PID2633175/create_time1789016590.7/command/boot, waits for that process
to finish, authenticates its exact result and starts the fixed fitting phase
once. Eight focused tests passed88830 in2.32s. No implicit retry/replacement.

The waiter has launched full-fit parent2636352/create_time1789018036.61.
Full-fit launchde9fb37f6cd51c609c21fb6ba03b2470167dedad2ecb9166907874bd351d1542,
same1144 sources,three workers,root go2_all_phase_matched_fits_v1_attempt_001.
Workers2636575/2636576/2636577 own the fixed direct/supervised_rollout/JEPA
round-robin assignments, six fresh fits each. Full caches are materialized;
the first seed's direct model reached700 updates and supervised/JEPA600 at the
last observed log point. No full eighteen-fit result is available yet. These
counts are progress observations, not final fit/admission claims.

Preserve waiter6946 and its original full-fit parent; do not manually launch a
second fitting phase. After completion, run full all-phase model admission and
then the prospectively defined training-only correction if its checks pass.
The main goal remains unmet:30 native episodes are audited, zero round trips.
