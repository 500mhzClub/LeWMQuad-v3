# Paired replay of empty retained-patch visibility skipping

Compare the completed ReceiptCopiedFootprintController with a separate
VisibilityBatchedFootprintController. Only the two initially empty historical
patch stores change implementation. Their coverage method computes one boolean
visibility summary per already-projected batch and omits visible-index
enumeration for empty rows. Keep the original prefix-key access, witness order,
early stopping, projection arithmetic and error fallback. The original map,
memory, selector, model, sensing and planning rules remain unchanged.

The synthetic microbenchmark preserved complete results. Empty and sparse
visibility patterns were about 29% and 26% faster, while fully visible and
immediate-success cases were slightly slower. These figures do not establish
a controller speedup or justify adoption without the real history comparison.

Admit the exact completed receipt-copy profile and preceding paired replay,
including original raw/model input bindings and all failure evidence. Use two
fresh models and controllers on the original 1,428 public observations, with
alternating execution order and no profiling. Require 1,425 forecast comparisons,
complete decision equivalence and the same seven retained-state identities.
Normalize only the new controller flag/identifier and the existing two patch
type paths in addition to the predecessor's declared normalization. Do not
normalize witness data, pixel bytes, maps, residuals or model state.

Use the unchanged paired replay body with private globals. Reauthenticate inputs
before and after. Preserve all output and failures in one exclusive attempt,
with no retry or resume. Assess hardware and live jobs before launch; use one
full CPU replay slot, at least 64 GiB available RAM, 41 GiB artifact space and
four physical CPUs. Keep deterministic single-thread settings and disabled
OpenCL. Do not modify any frozen native experiment.

Retain the original sensing failure at frame 1173 and failed round trip.
This is an incremental shared-host controller timing experiment, not new
physics, training, independent navigation, real-time or hardware qualification.
