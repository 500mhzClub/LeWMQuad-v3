# Ordered chained-anchor simulation waiter

This waiter owns one future invocation of the prepared chained-anchor native
launcher. It waits for the exact completed raw controller replay and the
existing contact-plus-flow waiter. That waiter is already ordered behind the
extended-budget and sustained-turn experiments and the preceding diagnostics.
It does not run a physics worker while either prerequisite owner is live.

The raw controller owner is PID 2840884, creation time 1789126982.74, launch
`8eba2f8dfea706109f8cec4fcf55206f36fa3b9f588c0bb2c344b492e95269cd`.
Its fixed completed result is
`d68e5e48916ff84d4034c95a2af5357695d277048519a15cf5dbb6a554703231`.
The contact-plus-flow owner is PID 2827789, creation time 1789121410.15,
launch `3fc8e765b6edc16120b418e6dc8cedf1da47eb134d959f98450b2adeaa2c6c72`.
Both exact argument vectors are source-bound in the waiter. It polls process
identity every 30 seconds and never treats an observation timeout as completion.
Ended prerequisites require an authentic completed result and source closure.
Failure, missing completion, changed completed identity, or a 48-hour wait
expiry ends this attempt without a retry or replacement.

After the final prerequisite ends, the waiter passes the fixed controller
completion-verification SHA-256 and the exact newly completed contact-plus-flow
waiter result SHA-256 to the native launcher. Full data, model, completed queue,
physics-idle and resource admission belong to that launcher. No alternative
model, controller, budget, layout, or scientific outcome is selected here.
The full native launch specification and interpretation remain in
`docs/go2_chained_anchor_maze02_pilot_v1_2026-09-11.md`.

After its single child ends successfully, the waiter authenticates the complete
new result and worker artifact roster, checks its original model and input
bindings, reconstructs the physical/public/controller prefix, and recomputes
the contact/timing/navigation readout. Zero round trips remains a completed
scientific failure when all technical checks pass. It never retries an
incomplete or failed child. No independent-layout policy selection, hardware
execution, sealed evaluation, or goal-completion claim follows automatically.
