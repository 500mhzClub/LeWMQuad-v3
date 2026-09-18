# Contact-scoring controller with measured tracking fallback: raw prefix V1

The original contact worker stopped at observation 561 because neither camera
supplied a current measured pose. The completed observer replay reproduced all
562 original visual records and recovered the current pose at frame 561 using
the unchanged direct-flow fallback and actual complete gyro history.

This experiment checks the full controller response to that pose. Instantiate
two fresh copies of the original assigned full-supervised model, state SHA-256
`755c074325af96d53649aba4927937113d8fab561ea05341a2f3c328598b2bb5`.
Run the original CommitmentContactAnchoredController and a separate subclass
that changes only its motion observer to DirectFlowDualCameraVisualMotion and
adds explicit controller/feature metadata. Keep the original contact-scoring
selector, floor registration, mapper, mission, residual logic, action bank,
motion vetoes, 3000-tick budget and sensing interfaces unchanged.

Reconstruct every original controller decision from actual raw packets and
compare its complete visual evidence to the fixed observer replay. The
candidate's complete visual evidence must also match that replay. Before
frame 561, require every candidate decision and forecast to equal the original
after removing only the declared controller name and feature flag. All current
visual and registered floor poses must pass their original public contracts.
Require public inputs to remain unchanged and every preceding requested command
to match the original completed command tape and physics endpoints.

Stop at frame 561 whether the full controller recovers or remains terminal.
If it recovers, require current floor evidence, a complete model prediction
and selection, and a requested command matching that selection. If it does
not recover, preserve its negative outcome and zero request. A recovered hold
still ends the replay: the original terminal drain is not a candidate future.
The original 558 preboundary forecast comparisons must all reproduce. Both
models must retain their exact states and have no accumulated gradients.

Authenticate the original ended worker and its complete raw artifact roster,
the original launch/model identity and the fixed completed observer result.
Use the actual assigned-model loader for checkpoint/correction validation.
This worker-based diagnostic does not claim completed parent admission or
rerun full training ancestry. A native follow-up still requires the original
completed queue and its existing full-admission rules.

Start only after the exact prior sustained raw and contact observer owners
have ended on the recorded boot. Use one CPU replay slot, single-thread CPU
OpenCV and BLAS, OpenCL disabled, 48 GiB available RAM including concurrent
native reserve, and at least 41 GiB artifact space. Limit output to 1 GiB.
Create an exclusive attempt directory; preserve failures and do not retry or
resume. No changed command is executed, no subsequent observation consumed,
and no native success, independent-policy selection or qualification follows.
