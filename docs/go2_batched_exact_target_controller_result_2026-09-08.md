# Batched map insertion: exact full-controller benchmark

The original and batched controllers exactly reproduced all 311 recorded
decisions (254 JEPA, 57 direct), including 283 selection frames. Fresh model
states remained unchanged. Only the six initially empty measured-bound indices
were replaced; observations, queries, collision semantics and control behavior
remained identical. This completed replay benchmark does not adopt batching
into a native experiment.

| Condition | Original selection median | Batched selection median | Median paired frame speed ratio |
| --- | ---: | ---: | ---: |
| Full JEPA | 575.095140 ms | 486.484166 ms | 1.169009302127432 |
| Full direct | 536.017695 ms | 464.200599 ms | 1.150226063769471 |

Every selection frame in both implementations exceeded 100 ms. These timings
cover controller.observe only, excluding packet reconstruction, model loading,
JSON serialization/comparison and native acquisition. The separate native
experiment's acquisition alone had a roughly 205-ms median. No whole-loop
speedup, real-time operation or navigation improvement is established.

One numerical thread/process ran alone, in original-then-batched JEPA order
and batched-then-original direct order. No second substantial job ran during
timing. This is one paired implementation pass per model, not a replicated
latency distribution. Total benchmark wall time including non-timed loading,
comparison and verification was 334.2196719760541 s. A process RSS sample
near completion was about 2.20 GB; terminal available RAM was 80,929,239,040 B
and artifact free space 62,574,915,584 B. No GPU computation was used.

Exact compact sorted-JSON decision stream SHA-256:

- JEPA: `e975fbd617e50b553b45ce7c20e01f72bfaa6b3b13bd2dabcf16688951d0d9a1`
- Direct: `b98774b6a3f3e1387084a35eef0e5866f1e067a3f79ad6c710ed24a163d2de23`

Under `go2_batched_exact_target_controller_v1_attempt_001` in the approved base:

- Launch: `312c4e919cb389eac144e11cfc547b3b51631fc50c4be7538f82280ec7c53e77`
- Result: `577a13a3331b00428e6ac75f2f4da47a3cb128095d8e5a9350b1a20aaa663a49`
- Bound native input result:
  `d9e0cef6a7e66459a5cc6d21cc6cf6e638666a33a12cdcfb8b2364f1477bd125`
- Source closure: 1,337 paths.

The optimization is a useful candidate for a separately declared future
integration. The next final-goal scoring experiment retains original indices
so its prospective behavior change has a single cause. The full novel-maze,
backtracking, baseline and hardware goal remains open.
