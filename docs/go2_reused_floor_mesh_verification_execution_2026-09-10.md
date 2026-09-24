# Independent mesh-reuse replay verification completed

The replay completed as
`1a7909f8ae1ab90a5e6281cfa187495140d029c384c7ea9d37422743854f543e`.
The independent checker passed, producing verification
`f9e9e615dbbb54ac98fc9b6563f8ac2c6b1f6ea8e1700a5c0763b83818f90820`.
The waiter terminal is
`4727dcd13a39c18faa93bc3921bd9768c1ec110ca2274a65b2c1af97a36637fc`.
Sessions 52230 and 70769 both exited successfully. Full findings are recorded in
`docs/go2_reused_floor_mesh_prefix_result_2026-09-10.md`.

The checker passes 29 tests in 2.26 seconds (session 10822). Source preflight
passed with 2,020 bindings (session 22056). The original replay's 2,017 source
bindings remain unchanged. The three additional paths are the checker, focused
tests and verification protocol; they are now frozen for this verification.

The raw replay ran as PID 2726828, creation time 1789062313.09, tool session 52230.
Its launch SHA-256 is
`15db8432621bbbc33fb098ce2a71bfa63b9b82fdf9e62421603508f8e879d6b0`.
That launch was witnessed while the exact owner was live and its source map
was checked against the original execution preparation.

The independent verification waiter ran as PID 2729358, creation time
1789063423.95, tool session 70769. It polled the exact original PID, creation time
and command every 30 seconds, with a three-hour timeout. After the original owner
ended successfully, it bound the final result hash and ran the checker with
the exact result and launch identities. It rechecked all verification sources
and recorded checker exit code zero and the final verification-document hash.
It did not restart the replay.

- Preparation: `docs/go2_reused_floor_mesh_verification_preparation_2026-09-10.json`,
  SHA-256 `6ead28e17b5d6d1bec85dc4d510b7e94f58822921f821cc7e8f7406488a0c4d4`.
- Scheduling: `docs/go2_reused_floor_mesh_verification_scheduling_2026-09-10.json`,
  SHA-256 `cea8734035f89eda1ef199f733b3491ed3fbc429a49eaf9b9d3401c7733214c1`.
- Terminal record:
  `docs/go2_reused_floor_mesh_verification_wait_terminal_2026-09-10.json`.
- Independent verification:
  `docs/go2_reused_floor_mesh_prefix_verification_2026-09-10.json`.

At scheduling, only observation 50 had been reported by the raw replay. The
completed verification now covers all 405 observations and all timing windows.
It adds no native episode or navigation success. The existing simulator queue
remains unchanged.
