# Paired retained floor coverage diagnosis V1

Authenticate the completed two-model bounded reobservation native probe and
readout. Replay all 110 recorded JEPA decisions with its fixed corrected model
and original controller, including actual auxiliary packets. Require every
decision and model state to remain exact.

At fixed ticks 62 (first auxiliary floor veto) and 99 (terminal repeated veto),
inspect every candidate's auxiliary floor-only foot intersection with no
other/unknown hit. Reconstruct the original full-foot centre and require exact
primary and auxiliary whole-patch witnesses. Test closed outward-rounded tilings
of the entire 44-mm square at 1, 2, 4 and 8 divisions per axis against primary,
auxiliary and both synchronized histories. Every tile needs a complete original
measured floor-pixel witness from one actual view. Preserve camera/frame provenance.

Both histories are queried read-only; no image interpolation, floor inference,
source-history mutation, radius reduction or unknown-return exemption is allowed.
If a combined 64-tile hold-foot query fails, explain the failure with the original
frustum and floor-pixel checks for each tile. The virtual auxiliary reference pose
already stored by the original controller preserves its actual optical transform.

This is a diagnosis, not an adopted controller change or a new native outcome.
Use one CPU thread for the single fixed case, with 8 GiB available RAM and 256 MiB
output allowance above the 40-GiB storage reserve. Freeze sources and input
bindings before exclusive root `go2_paired_retained_floor_coverage_v1_attempt_001`
and reverify afterward. Preserve all failures and report incomplete coverage.
