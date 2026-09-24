# Hold reorientation: saved-selection boundary result

The first changed request is observation **405**: hold `[0, 0, 0]` becomes
left turn `[0, 0, 0.45]` after the ten preceding hold requests at 395–404.
All complete saved selections before 405 remain identical. At 405 only action,
action index, requested command and the explicit intervention receipt change.
The raw forecasts, model utilities and all geometry vetoes are identical.

Left turn is the only eligible turn at the boundary. Its original utility is
−0.004885133973799164 m versus holding's +0.002772580716459812 m. This is the
declared willingness to reorient despite lower immediate model utility, not
evidence that the turn improves the physical outcome. Observed goal distance
at the boundary is 3.353074019598158 m.

Output: `go2_hold_reorientation_saved_prefix_v1_attempt_001` under the navigation
artifact root. Result SHA-256:
`ea832425482ec3f82e3a91407b5a9aa0029f1bad466213e64b8c83cce18f651c`.
Launch SHA-256:
`c874464f23c923bf314371c38898fc78a52d3daababea84eec8df6e27a96fc5f`.
There are 1,938 bound sources and three bound output artifacts. Tool session
37518 exited zero. Independent verification in session 11275 exited zero:
all source/output hashes passed, all 406 comparison rows reconstructed, the
consumed original-prefix canonical hash matched, and the saved first boundary
exactly matched the state-machine output.

Focused tests: session 3763, 22 passed in 1.85 seconds. Coverage includes the
ten-prior-hold boundary, each original veto, unchanged evidence, absent turns,
fresh observation requirements, resets, model ranking, invalid inputs, the
actual selector delegation and unchanged inherited observe/advance methods.
The selector delegation test uses a mocked original choose method; it does not
prove raw model/controller compatibility.

No raw sensor reconstruction, model inference, candidate command execution or
post-intervention observation occurred in this check. The original first case
has collected 1,529 observations and stopped with no feasible candidate after
1,528 completed commands and ten terminal drain commands; its full worker audit
was still pending at 10:06 UTC. That collection is not yet counted as a newly
completed audited episode.

Next: admit the completed original first-case worker and its full raw audit,
then reconstruct both full controllers from raw RGB/depth/body/control packets
through observation 405 using fresh identical expanded JEPA models. Require
every complete original decision and forecast to reproduce, unchanged observed
map/contact/mission/executed-residual history, exact candidate agreement with
this boundary, and no observation 406. A prospective native test must wait for
the already queued frontier experiment and be separately frozen; neither the
active batch nor that queue has been modified. No independent layouts have
been consumed by this work. The overall navigation goal remains unachieved.
