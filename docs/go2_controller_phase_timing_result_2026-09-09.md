# Full-prefix controller phase timing result

Session8699 CLOSED exit0. Root go2_controller_phase_timing_v1_attempt_001,
result434397c65391d235d6708a3e24675118463aea6ed89e9061721e71e643a0ce0a,
launch2b6921e493f6bafe92065badc5d4aaec25fb0c9c453cc5ce4c29cd2e4eb35d9b,
streama2ec0dfeee8e91c718aa59b4bd4057438d610a5677ff533c23bfad1f93127e5f.
1535 sources,960 complete decisions exact, model state unchanged, hooks removed,
892.8876304258592s wall time. Six instrumentation tests previously passed3.65s.
Input/source/model checks completed; the final result and stream hashes were
checked again when reading these aggregates.

Median instrumented controller duration839.349784ms against100ms decision
intervals. Median model forward7.3161635ms; median motion observation50.0884295ms;
median floor registration73.3845145ms; median map observation367.2935825ms;
median action selection309.2054515ms. These are inclusive per-phase medians
and must not be added as a partition of the median total.

Mean exclusive costs identify substantial nonmodel work: selector190.815819ms,
contact queries130.224387ms, original auxiliary observation117.933796ms,
primary insertion90.887013ms, floor registration75.496551ms, primary
classification65.632869ms, primary coverage58.522121ms and motion53.670113ms.
The model mean is7.430662ms. Every frame's exclusive durations sum exactly
to that frame's root duration. Instrumentation overhead is included. This is
a phase diagnosis, not a controlled speed comparison or real-time result.

No speedup implementation was installed from these measurements. Future
optimization should address measured geometry/map/selection costs and require
complete-decision equivalence plus an appropriate controlled benchmark.
