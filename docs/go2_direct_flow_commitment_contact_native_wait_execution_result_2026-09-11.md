# Contact-plus-flow physics handoff is running

Prepared and started one waiter for the contact-plus-flow maze 02 pilot. All
40 focused tests passed in 2.52 seconds; source preflight verified 2,208 source
bindings. Hardware assessment passed with about 76 GiB available RAM and
567 GiB free on the artifact volume. No physics scene was started.

The exact waiter is PID **2827789**, creation time **1789121410.15**, with
command `.generated/venvs/genesis_rocm_0_4_6_v1/bin/python -B
scripts/await_go2_direct_flow_commitment_contact_maze02_native_v1.py`, boot
`1264d80f-6e46-4fcd-b2fd-2a5d7b964c73`. Tool session: **33875**.

- Waiter launch SHA-256:
  `3fc8e765b6edc16120b418e6dc8cedf1da47eb134d959f98450b2adeaa2c6c72`.
- Preparation record: `go2_direct_flow_commitment_contact_native_wait_preparation_2026-09-11.json`,
  SHA-256 `4387ed534f5da2a6c5d47f4c8fa67b07ab84c7973a7c033cfc4c5fcf96f0fcb6`.
- Verified process/start record: `go2_direct_flow_commitment_contact_native_wait_execution_2026-09-11.json`,
  SHA-256 `f2203b2980cb880fbeef88a4bd62596c854cedbd47b49b0522b290cc9c2e5b1c`.

The first verified event records the contact controller replay as completed at
its exact result hash and the original sustained-turn waiter as still live.
The new waiter had no child, and its native output was absent. It will wait
for the original queue, start one prepared child, and authenticate the actual
completed physics result. Failure or timeout causes no retry.

At the process recheck, the original tracking child PID 2824049 was live with
advancing CPU and read counters (about 412.5 GB read), and no scene child yet.
The original tracking parent, extended-budget waiter and sustained-turn waiter
were all live. Keep their exact definitions and queue order unchanged.

The new handoff is operational; a contact-plus-flow physics outcome is still
pending. No round trip, independent-maze benefit, real-time qualification or
hardware qualification has been established. Revalidate process identities
and completed results before reporting later state.
