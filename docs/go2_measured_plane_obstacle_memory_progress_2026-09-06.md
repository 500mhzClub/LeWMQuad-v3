# Measured plane obstacle-memory prototype

Implemented a separate, unlaunched development consumer in
`lewm/measured_plane_obstacle_memory_development.py`. No recorded physical
experiment, frozen source, threshold, or result was changed. Full maze results
remain 0/2; there is no new learned-navigation or JEPA result.

Each immutable prepared depth frame now selects its median row-major eligible
plane cell once, at observation time, independently of current foot queries.
The hypothesis binds depth/validity hash, gravity-up vector and selector policy;
it is retained and evicted with the causal frame memory. An eligible plane is
not a claim that a surface is dominant ground or traversable. No eligible cells
means no hypothesis, not fabricated floor support.

The new consumer partitions primitive queries before applying plane-family
exemptions. A supplied physical-gap interval must prove separation, or straddle
zero for one of the four exact foot primitives. Non-foot intersections and
definitely submerged primitives retain near-return vetoes even in partial views.
Positive clearance and foot-contact candidates still require complete measured
floor footprint coverage and non-floor evidence. Contact permission, future-gait
qualification, calibrated error bounds and navigation qualification remain false.

Focused execution 51414 initially passed 12 tests and failed two assertions:
the proposed floor-only counterexample used a room fixture containing side walls,
which already supplied an obstacle veto. The fixture was corrected to an analytic
floor-only depth image; no implementation predicate or assertion was relaxed.
Execution 87969 then passed all 65 tests across the new 14-test file and the two
existing primitive-obstacle/beam-kernel test files in 6.10 seconds. The regression
reproduces the old global-mask veto suppression and verifies that per-primitive
gating restores it with both reference and compiled backends. It also covers
identity, fault latching, missing hypotheses/cells, partition ordering and parity.

This is focused synthetic verification only. It has not yet been evaluated on
the recorded Go2 packets, run through the expanded regression suite, integrated
into a prospective controller, or tested in a new physical mission. Both test
handles are terminal; no training or simulation job was launched.

Next: compare the new consumer on the already recorded 26 bounded-floor Go2
observations with binding verification before and after. Measure actual floor
coverage beneath the robot rather than assuming the forward camera observed its
starting support region. Resolve startup observability and explicit modelled
foot-contact admissibility, then prospective motion/full-loop latency and fresh
complete discovery/return missions. Matched JEPA/supervised training, memory and
genuine multistep planning comparisons on independent layouts remain required.
