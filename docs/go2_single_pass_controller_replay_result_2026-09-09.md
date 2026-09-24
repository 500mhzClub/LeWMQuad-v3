# Single-pass persistent-query controller replay completed

Result: `4687d67fbb53fce3b29a122e379b51805fa685b4c39e3e817e3a5862374d342e`.
Launch: `f3bfad77ee427563b1d94a1090308929486248bcbd63e3ab67bd857b9f2611b9`.
Stream: `649b85ece2777a66931465e4d65d8dc9a09c9152c70ccb9182d7fdaf0666e07c`.
Root: `go2_single_pass_maze_controller_replay_v1_attempt_001` in the
development navigation artifact store.

All 1881 complete decisions match the ninth original episode, including its
first terminal failure at 1870 and the zero-command drain. Model state is
unchanged at 4f796afdf32c2c6dbcd1d5981834a7b17be86e2efdafd9427b2d80240def0ca6.
The final source/input checks pass; 1554 sources. Wall time 1784.984079356 s
is not a controlled speed comparison.

SinglePassLaterFloorController replaces the eight persistent sample-bound
indices with packed-owned insertion and single-enumeration query predicates.
Current-frame indices and complete decision labels are unchanged. Component
queries previously reproduced all receipts at approximately twice the query
throughput; that workload was not the full controller query distribution.

This completed replay admits the prepared 256-frame alternating-order paired
controller benchmark using this actual result hash. It grants no native
adoption, whole-loop speed claim, real-time qualification or improved
navigation outcome. The tenth settling experiment uses its original frozen
mapping implementation. The full unseen-maze goal remains unachieved.
