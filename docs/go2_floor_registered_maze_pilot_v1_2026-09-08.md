# Prospective floor-registered maze pilot V1

One fresh CPU native scene on reused development maze 0 tests the separately
named floor-registered controller. This changes the online pose estimate used
by both camera maps, retained contact evidence, residual scoring and mission
tracking. It preserves the independent raw visual observer, learned model,
commands, physical mission, robot geometry and strict visibility evaluator.
The static shared floor remains a hypothesis; uncertainty and support are not
certified. No prior failure is relabeled.

The exclusive output is `go2_floor_registered_maze_pilot_v1_attempt_001`.
`scripts/run_go2_floor_registered_maze_pilot_v1.py` requires the exact completed
prefix-result SHA-256. That result must reconstruct the complete confirmed-floor
predecessor and preserve the original visual witness, with no model updates.
It must stop at a nonterminal changed command. Candidate admission failure,
no intervention or a changed terminal stop is not sufficient to launch motion.

Before creating output, verify the prefix, completed predecessor/readout,
model inputs and source bindings. Run `--preflight-only` to inspect current
hardware. Require 32 GiB available RAM and the unchanged 10 GiB collection plus
1 GiB persistence allowances above the 40 GiB artifact reserve. One native
scene worker and one thread per BLAS/OpenCV/PyTorch operation are used. No
parallel native scene is planned. The launcher records resources throughout;
these admission checks are not OS-enforced quotas.

The collector runs the existing outbound/return mission with the shared 3000
navigation-tick budget and ten-command terminal drain. All physical, contact,
actuation, acquisition, nominal planning and measured-floor requirements remain
active. Plane registration failure stops the candidate without substituting a
native pose or extrapolating a measured floor.

After collection, reconstruct every public observation, complete controller
decision and command with a fresh model/controller. Preserve both raw and
registered pose evidence. Evaluate actual arrivals, traversals and returns, and
retain the unchanged primary/auxiliary visibility checks. Compare physics and
public packets with the predecessor through the intervention observation.
Every new controller decision in that prefix must also exactly match the
prospective replay stream. Do not attach the predecessor's post-intervention
outcomes to the new command.

The original confirmed-floor result is
`5ee4ef051e1a506f205aae51610deece18f755fb5440c40b1113e22e5ba317ee`.
It failed navigation and strict visibility at frame 909. If the new command
changes before then, this pilot generates a different subsequent trajectory;
its own complete visibility audit is required, with no inferred pass.

This pilot adds no independent layout by itself. No retries, output overwrites,
source changes after launch, training, checkpoint selection, real-time or
hardware qualification are included. Completion of this pilot does not complete
the thread goal; verified missions on independent mazes and matched comparisons
remain necessary.
