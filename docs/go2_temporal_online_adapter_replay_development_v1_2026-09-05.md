# Relative orientation and temporal online adapter replay V1

This package checks the new sensor-only coordinate tracker and repeated temporal
adapter against existing audited development observations. It performs no fitting,
new simulation or physical commands. It does not revise the negative JEPA result.

The tracker starts at the declared control boundary with identity relative
orientation and integrates each new 50-Hz body gyro once using midpoint SO(3)
increments. It requires consecutive 100-ms packets and consistent overlapping
sensor history. Failure latches; stale orientation cannot be reused. There is no
turn-specific 12-second deadline, translation estimate, absolute heading, bias
correction or hardware accuracy guarantee. A supplied initial-body planar
direction is rotated into the current body using the transpose of the estimated
initial-body-from-current-body rotation. Near-vertical projections are rejected.

The adapter receives every100-ms RGB/body/control packet, warms a four-frame
history and starts its direction at the explicit control boundary. Every .5s it
evaluates the first .5-second block of all five fixed actions; other blocks are
unknown/masked. All three fixed seeds are used for each learned arm, with average
probabilities rather than average logits. Direct and rollout heads stay distinct.
The score is10×contact probability plus distance to the transported .8-m cue.
The cue is directional, not a fixed point goal. Each output contains exactly five
requested/expected-applied commands, all candidate predictions, source/model/input
bindings, orientation and timing. Invalid input or selection latches failure;
the future physical caller must explicitly issue a safe command.

## Fixed replay population and checks

1. Use all40 original validation branch streams and six methods (all stop and
   the five condition/head combinations). Replay each recorded stream to its
   terminal command-clock observation, selecting at its available derived-window
   boundaries. Use initial direction(.8,0). The actual data's 304 windows imply
   1,824 proposed choices; their commands do **not** determine subsequent replay
   observations. This is off-policy input/output replay, not closed-loop evidence.
2. For the recorded continuation candidate only, compare each of three model
   outputs with its bound offline first-horizon prediction, tolerance1e-5 for
   batch-size floating-point differences. Exclude noncanonical initial sibling
   images explicitly; never replace live RGB with training reference RGB. Every
   later context is its actual own branch. No accuracy label is invented for
   an unexecuted switched candidate.
3. Compare the orientation estimate against separately read, hash-bound physical
   orientations on those40 streams, once per physical stream (not once per
   method). Also replay all18 fixed gyro-turn arena streams. Maximum SO(3)
   angular error and projected-heading error must each be at most .04rad for
   this ideal-sensor development check. The limit is fixed before replay, and
   failure is retained rather than adjusting it. Clock and episode errors are
   integrity failures, not heading-error samples to omit.
4. Record commands, input hashes, predictions, compared/excluded populations and
   source/input/result identities. Verify source stability. Synthetic tests cover
   a20-second nonplanar integration, coordinate handedness, drop/reset/rewrites,
   ensemble population, probability averaging, candidate slew, cadence and faults.

The numerical replay tolerance validates plumbing, not prediction accuracy.
Passing the orientation threshold only covers these ideal recorded streams; it
does not qualify sensor bias/drift or narrow-clearance turning on hardware.
Next is the separately fixed fresh physical successive-decision protocol, with
native stopping, all-trial accounting and explicit action-switch diagnostics.
