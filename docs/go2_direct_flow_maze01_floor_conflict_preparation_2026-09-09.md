# Preparation: reconstruct the completed tracking pilot's floor rejection

The final maze-1 tracking result is
`d6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de`.
The completed raw audit and physical-prefix comparison pass, but frame 504
rejects a floor candidate against the transported reference. The diagnostic
requires this final result and uses its verified saved witnesses, never an
interim collection as sufficient admission.

The prior successful receipt completely records the original registration's
reference, admitted anchor and previous frame. The terminal decision records
its original raw visual evidence even though registered evidence is null.
The new helper restores that original registration state and calls the unchanged
method on actual paired packet 504. It requires the same exception and unchanged
anchor/reference/frame, with failure latched. Original candidate extraction and
composition reconstruct every current camera residual, including camera counts,
signed extrema, RMS, worst point/pixel, above-3-mm counts and cloud/mask hashes.
No candidate trimming, threshold change, model load, tracker rerun, native truth
or following observation is needed. Every input and saved witness is checked
for mutation. No subsequent action outcome is inferred.

Validation: 22 tests passed in 4.11 seconds, session 18181, exit 0. They exercise
actual synthetic depth clouds with full-plane and transported predecessor states,
reproduce the unchanged rejection, reject stale/forged witnesses and invented
conflicts, account for signed residuals and pixels, exclude decisions after the
first terminal, and require the completed raw audit and physical prefix.
The first test run (55747) had two fixture errors because live tuple identities
were passed to the explicitly JSON-input helper. Tests were corrected to use
the same serialized form as the real decision stream; production code and its
strict JSON restoration were unchanged.

Frozen sources:

- `lewm/floor_transport_conflict_readout_development.py`:
  `24ac6fb03e10503f6f658c1edba113f7af908856e5ad08c34a9e77dd99e479db`.
- `scripts/diagnose_go2_direct_flow_maze01_floor_conflict_v1.py`:
  `167a128877b324c3eafcd23ae21e8e3a35e862528736b52ee3f4930ecc7a4048`.
- `lewm/tests/test_floor_transport_conflict_readout_development.py`:
  `b4ae56283d5f10debf06dd55f81718771033e6ff3fc18970bcbc96a61f89e09a`.
- `docs/go2_direct_flow_maze01_floor_conflict_v1_2026-09-09.md`:
  `334520cdbd7dd1a4ce2605a7d8b06c564effb67d05892fe4258442c992b9d3b4`.

Fresh hardware 78334 exited 0: 16 physical/32 logical CPUs, all 32 affinity,
CPU 6.3%, 81,062,461,440 available RAM bytes, 75,988,107,264 artifact-free bytes
and 21,358,866,432 workspace-free bytes. Both GPUs idle. The native parent and
worker have ended. Only the independent phase diagnosis was active. Two new
8 GiB CPU analyses plus that diagnosis's 8 GiB allowance fit comfortably.
Each new runner refreshes its resource checks after full input authentication.

Actual completed-result admission was checked separately (43789, exit 0), with
launch, audit, prefix and worker-terminal hashes checked against the final result.
Submitted floor diagnosis session 14291 with the exact native result SHA and
single numerical threads. It remains subject to full source/input admission;
submission is not a completed result. Preserve the exclusive output
`go2_direct_flow_maze01_floor_conflict_v1_attempt_001` and any failure.
