# Tracking-recovery native-prefix comparison prepared

The future native comparison now has explicit admission and physical-prefix
checks in [the checker](../scripts/no_rgb_jepa_direct_flow_native_prefix_development.py).
It can admit only a completed positive full-controller replay with the assigned
no-RGB JEPA state and the exact frame-859 intervention. It reconstructs all 860
saved comparisons, checks the original command endpoints and raw public packet
fingerprints, and rejects incomplete, changed or negative evidence.

For a future fresh episode it checks all 43,700 pre-intervention physics samples,
all 860 public observations, all 859 preceding commands, and complete candidate
decisions against the prospective replay. The intervention command itself must
complete. A recovered measured hold is explicitly distinguished from movement;
following physics is not required to match the original terminal trajectory.

Verification completed: **53 synthetic tests passed in 3.73 seconds**. These
include altered physics, truncated streams, wrong observations, changed model
forecasts, wrong command endpoints, incomplete intervention execution, missing
artifacts, source mismatches, wrong model/case and negative controller results.
Admission tests mock digest/file reads; they do not establish real positive
native admission. Real source verification checked and froze 1,951 paths.

- [Source and execution preparation record](go2_no_rgb_jepa_direct_flow_native_prefix_preparation_2026-09-10.json).
- Preparation SHA-256: `0968a0b072954e050d6d27450776a6d614662fbaecb07b0070dee2c4bd616333`.
- [Prospective comparison scope](go2_no_rgb_jepa_direct_flow_native_prefix_v1_2026-09-10.md).

No native launcher or collection was created or executed. At preparation, the
original full-controller prefix owner (PID 2749113, creation 1789073975.55) was
confirmed live in original input admission, and the fifth matched native case
was still running. Neither quiet hashing nor an absent output root is a reason
to restart that owner. A positive completed prefix is still required; its
negative result, if any, must instead guide diagnosis.

Next review the full-controller result. If positive, finish the narrowly scoped
collector, full raw audit, queue-completion admission and one-episode launcher,
using this checker to establish the real intervention. Preserve the existing
six-case → frontier → hold → contact native ordering. Review the other queued
outcomes before finalizing the independent eight-layout/four-method population
policy. The prepared paired timing replay also remains unexecuted and must wait
for the current full-controller replay owner to finish.

This work prepares the transition from recorded observer recovery to a fresh
physical simulation test; it establishes no new navigation outcome. The goal
remains active, with 41 completed audited development episodes and zero verified
round trips at the last completed-episode audit.
