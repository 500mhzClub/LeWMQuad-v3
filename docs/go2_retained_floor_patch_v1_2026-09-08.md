# Retained measured-patch foot coverage diagnostic V1

Authenticate the completed measured-floor-contact native cohort/readout and
reconstruct every available direct-039 map and classification receipt. Preserve
every original six-candidate surface/contact check. No native execution or
command change. Retain unavailable observations and all original failures.

For each past/current valid observation, retain an exact int32 summed invalid-
pixel index using the existing four-pixel ground-normal/planarity tests and
10-mm measured-plane height band. Bind its RGB/depth/frame identity and observed
pose; never retain a future frame. Capacity is 4,096 frames with no eviction.
For each predicted foot, project its enclosing 44-mm floor-plane square into
each retained frame. Require the ENTIRE outward-rounded pixel rectangle to
pass the same measured tests in one frame, including range and image bounds.
Missing pixels remain unknown; no union interpolation or plane-only coverage.

Compare this footprint-specific witness with the original 5-cm grid requirement.
Preserve original positives; a new conditional coverage result may use grid
coverage OR a complete retained patch. Keep all non-floor/unknown foot conflicts
and all non-foot conflicts. Save every witness and the remaining candidate
conflicts without changing the controller. These nominal tests do not certify
terrain interpolation, pose uncertainty or physical support.

Five focused analytic tests must pass. Inspect hardware; require 8 GiB available
RAM, 41 GiB artifact free space and 1 GiB output capacity. Process this one causal
history sequentially. Benchmark the first 40 frames with batched 24-foot queries
versus separate single-foot queries, require exact results and choose the faster
query mode. Preserve failures in an exclusive terminal attempt, without retry,
model fitting, old-result mutation, promotion or hardware qualification.
