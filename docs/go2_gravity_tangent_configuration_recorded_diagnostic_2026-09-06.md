# Exploratory gravity-tangent configuration diagnostic

The preceding fixed body-axis configuration queries found front-foot/lower-leg
vetoes at 0.75 and 1.00 m. This separate diagnostic asks whether pitching the
translation with the measured body axis contributes to those negatives. It is
exploratory and motivated by that result; it is not an independent comparison or
a replacement/rescore of the original five configurations.

Replay the same bound 15 startup/tail observations without physics or controller
restart. Use the existing `gravity_basis` with the current sensor-derived up
direction, and translate the unchanged measured posture by 0, 0.25, 0.50, 0.75
and 1.00 m along its forward tangent. Record the two direction vectors and their
up components, evaluate every configuration with both reference and compiled
consumers, and retain all residual unknowns, whole-query vetoes and ground
candidate results. Use the same 3.3-s query horizon, unchanged 3.5-s setup expiry,
pose proxies and geometry/plane/range allowances.

The tangent is a geometric convention based on observed gravity, not a learned
command response, ground-contact model, trajectory or validated swept volume.
No physical action, new maze result, JEPA contribution or hardware claim follows
from this diagnostic. No existing source/result/protocol is changed.
