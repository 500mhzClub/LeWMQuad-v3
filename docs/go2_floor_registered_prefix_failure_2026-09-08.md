# Independently fitted floor-plane registration prefix failed

Session 43443 exited 1. The exclusive
`go2_floor_registered_maze_prefix_v1_attempt_001` is terminal and must not be
restarted. Its launch SHA-256 is
`bfcf3260e39a15ec7d94858b492099e85fde2349fbb11eeddc2aafd348a5b92e`,
failure SHA-256
`562acbc1463ef16864be4e6cf2128e47289d2441258585b4e521799b6fe53c81`,
decision-stream SHA-256
`542e85af02532711c6777ffb8ef7fd0f20360274459de35d793cc07b6f006889`.
There is no successful result and no authorized-by-prefix native intervention.
The prepared `go2_floor_registered_maze_pilot_v1_attempt_001` was NOT launched.

The candidate matched executed commands through frame 328, but registration
failed at frame 329: `two current independently admitted measured planes required`.
The replay requires valid candidate admission and therefore retained this as a
failure, not a passing first-command intervention. Its collector, controller,
geometry, evidence, tests and protocol bindings remain frozen.

Read-only diagnostic session 98438 completed on public packets from the completed
confirmed-floor native result
`5ee4ef051e1a506f205aae51610deece18f755fb5440c40b1113e22e5ba317ee`.
It inspected frames 0, 327, 328, 329, 330 and 331. The later frames are descriptive
predecessor geometry only; they are not an unexecuted continuation of the failed
candidate. At frame 329:

- Primary: 1179 candidates, second covariance eigenvalue
  0.0020401083831666457 m², below the 0.0025 m² independent-plane gate.
  Maximum residual 0.00000448656238249967 m. Rejection reason:
  `insufficient_two_axis_extent`.
- Auxiliary: 13303 candidates, second eigenvalue 0.022449931065491948 m²,
  maximum residual 0.00000548167256941845 m; independently admitted.
- Primary patches at frames 330 and 331 also fail independent two-axis extent,
  while auxiliary patches remain admitted. This is a narrow forward view,
  not absence of a well-spread measured floor in the combined observations.

The successor hypothesis fits one common plane to the combined current camera
measurements. Combined rank must pass the unchanged numerical extent threshold;
every contributing point must still satisfy the unchanged 3 mm coherence gate.
No individual sample or opposing surface may be discarded to pass. This is a
change in the estimator's measurement model, not a retrospective revision to the
failed independent-plane protocol. It requires a separate source freeze and
fresh prospective replay. No navigation, arrival, return or hardware success
was established by this failed prefix.
