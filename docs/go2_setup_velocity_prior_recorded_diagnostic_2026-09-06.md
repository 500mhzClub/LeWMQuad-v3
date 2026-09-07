# Fixed conditional setup-prior diagnostic

Scope: read-only algorithm diagnosis on the existing 26-frame bounded-floor Go2
interface tape. This does not rerun/rescore its physics or qualify a maze result.
The initial-velocity condition is explicitly proposed after that tape exists;
this is not a preregistered independent physical validation of a chosen bound.

Use a zero initial-body velocity mean and a Euclidean radius of 0.02 m/s at the
first sensor epoch (1.5 s), episode (0, 0, 0). The protocol hash is an immutable
reference to this proposed condition, not proof that the condition holds. Keep
the original no-prior frame-1 failure. Do not tune the radius or extend the
inherited 0.08-m development scale budget based on results.

Propagate the supplied velocity ball through the recorded depth weak subspaces
and causal gyro rotations. Add its position radius to, rather than replacing,
the inherited uncalibrated error scale. Test that actual historical ray memory
uses that combined scale and stops when its budget is exceeded. A separate
offline integrator may report all 26 frames; any output after the memory stop
is diagnostic only. Never restart or re-enable stopped memory.

After computation, use the already recorded raw evaluator trace to report
whether the proposed initial velocity ball actually contains the reference
initial velocity. Report position errors against the relative evaluator body
trajectory separately. Evaluator poses, velocities and contacts must never be
passed into either policy packet, integrator or memory consumer.

No starting-region prior is injected in this diagnostic. The separate finite,
expiring region contract only defines conditional non-floor clearance. It does
not establish supporting ground, foot contact, prospective gait safety or a
verified startup. An independent setup-validation procedure and a real
contact-aware controller remain required before a new full-mission experiment.

Verify existing launch/result/reader identities, inherited bound sources/inputs,
raw artifacts, and explicitly listed development source/test/protocol hashes
before and after. Emit diagnostic output without modifying the recorded tape.
