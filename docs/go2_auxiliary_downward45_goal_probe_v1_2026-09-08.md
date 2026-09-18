# Explicit 45-degree auxiliary camera native goal probe V1

Run the fixed corrected seed-2026091001 JEPA/direct pair on reused development
layout family_episode_039, in unchanged seed-2026091501 order. Require full
correction admission and readout, the completed 30-degree reobservation native
probe/readout, the audited 18-frame 45-degree native sensor capture, and two
exact fresh 45-degree controller replays per model with all frames admitted
and unchanged model state.

Change auxiliary pitch from 30 to 45 degrees with explicit calibration identity,
native capture, optical/body conversion and retained floor-patch transform.
Retain every valid return and original occupied, other/unknown, full-foot
coverage, articulated collision and 0.45-m nominal checks. The primary observer,
model inputs, weights, training-only correction, candidate bank, eight-step
planning, view/route logic and bounded ten-command reobservation are unchanged.
Native pose and segmentation remain confined to renderer/evaluator use.

Preserve the 240-navigation-tick budget, warmup, arrival dwell, terminal drain,
actuator slew, gait, gains, physical stops and original native 0.06-m goal gate.
Zero commands carry no clearance guarantee. Count a success only after all
native goal and primary/auxiliary measurement gates pass. Replay every online
decision from the recorded public packets with a freshly loaded assigned model.

Freeze complete source and input bindings before exclusive output root
go2_auxiliary_downward45_goal_probe_v1_attempt_001. Use two independent CPU
scenes and one numerical thread each, with prior scaling evidence and a fresh
hardware check requiring 32 GiB RAM and 8 GiB output allowance above the
40-GiB reserve. Monitor resources and whole-iteration wall time. Require exact
first-four paired public frames and first-900 native samples against the
audited 45-degree sensor capture. Reverify all bindings afterward and preserve
both outcomes and any failure without retry or resume.

This is a prospective development probe on one reused integration layout.
The preceding capture and shadow replay establish neither navigation nor
counterfactual goal arrival. Physics remains paused during computation.
Independent mazes, exploration/backtracking, matched reactive/nonpredictive
and planning/memory comparisons, realistic timing and bounded hardware evidence
remain outstanding requirements.
