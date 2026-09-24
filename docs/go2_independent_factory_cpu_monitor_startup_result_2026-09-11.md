# Real factory startup passed CPU monitoring

All four fixed factory/controller paths completed under the CPU monitor with
OpenCL disabled before interpreter import. All 16 saved decision rows and all
four complete factory reports matched the original completed startup exactly.
Every monitored scope completed without a recorded violation.

Result SHA-256:
`d7810ed1ad163a5c490cdf2302af8678bca82f71e0414d122cc89dce4b8794f3`.
Launch SHA-256:
`fd1d226e0364400fec17abcbde792b81d51ac1b4770dc68aa139fa9717665e69`.
Output root: `go2_independent_factory_cpu_monitor_startup_v1_attempt_001`.
Original owner PID 2769476, creation 1789084988.86, ended; tool session 21960
exited 0. No retry or replacement was started.

| Arm | Packets | Recorded CPU tensor operations | Bounded case wall time |
| --- | ---: | ---: | ---: |
| Persistent JEPA | 4 | 10,792 | 54.93 s |
| Persistent supervised | 4 | 10,792 | 54.53 s |
| Reactive | 4 | 78 | 51.62 s |
| Current-pair JEPA | 4 | 10,792 | 54.85 s |

These times include model/factory construction, the original paired startup,
state checks and saved-output comparison. They are not controller latency or
an isolated timing benchmark. The reactive path still has no assigned
high-level model; CPU tensor-operation counts are not a proxy for model use.
The profiler recorded approximately 41.7–44.5 million Python calls per arm;
its cost in the complete raw auditor remains to be assessed.

The original JEPA state `35496b6b402013f7a33d9c30110e115b90ded672f0d0da829a07128601507b6a`
and supervised state `755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`
were unchanged. The original deliberately incompatible factory behavior,
adapted forecasts, observed-state checks and first planning requests all
remain identical to the reference. Both original and adapted factories use
the already specified study public mission while reading old development
startup packets. This is not reconstruction of the old native mission, new
independent-layout data or physical navigation. No packet after the first
planning request was consumed and no new command was executed.

Completion authentication checked 2,041 source bindings, nine outputs, the
original reference/input bindings, all four monitor records and all 16 saved
rows. Verification session 97204 exited 0 and did not rerun inference:
`docs/go2_independent_factory_cpu_monitor_startup_completion_verification_2026-09-11.json`,
SHA-256 `6130e28a098686c5145b14643f96c6c070ce2f231d6d6fd9b4bc490d49acff87`.

Preparation used 29 focused tests, session 71255, exit 0, in 4.96 seconds;
source preflight session 84511 exited 0. Preparation record:
`docs/go2_independent_factory_cpu_monitor_startup_preparation_2026-09-11.json`,
SHA-256 `8bc7e9561187c44b26cb76ea761f74ad3bd0b4c6ce00ac74122919e770a44b96`.

This establishes bounded real factory compatibility with explicit CPU
instrumentation. Full raw-auditor qualification, monitoring integration into
the process driver and source-bound overlap admission remain incomplete.
Independent study execution still awaits final policy and queue/input review.
