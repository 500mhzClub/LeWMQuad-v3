# Marker V1 audit correction: native box data padding only

The six-case physical acquisition completed in session 88299, exit 0. The V1
audit (43351, exit 1) failed before completing the first trial: the collector
correctly stored all seven native `RigidGeom.data` values, but the static-object
checker compared that array directly with three box dimensions. Its synthetic
fixtures had modeled only three values and missed this native encoding detail.

Installed Genesis 0.4.6 `rigid_geom.py` lines 102–104 initialize seven zeros and
copy supplied geometry data into the leading slots. The primitive builder sets
the box data to its three extents. The collected rows contain those extents plus
four zero padding entries. Installed `rigid_geom.py` SHA-256:
`501d5ce71ab2249d5971e501f003fac9ecbcb96db5ad893e92ba9ab850655d8a`.

Preserve the original audit FAIL and all original source/test/protocol bytes:

- Launch: `9514555d90fc360d07e0c7efe29afe8aa9047ddbab57295019e4a8dae321faa7`.
- Physical result: `4300c237d9ff9ba3677ea58bc904faa56f42089e03a01e9ca3dd29e84520c019`.
- Original failed audit: `1a99c1ce9ede3712849d26b554c1a5804954781dcab94cec53976ff020ccc2b9`.

The separate V2 auditor requires exactly seven finite values, exactly four zero
padding values, and the same 1e-7 absolute/zero-relative tolerance for the first
three extents. All other static-object checks and the full trial replay remain
source-identical. Tests compare the source bodies and reject short/long data,
nonzero padding, nonfinite entries and changed extents. The V1 failure remains
reproducible. No physical, perception, expected-response or success criterion
changes; no new RGB, physics, detector inference variant or threshold fitting.

One corrected audit may read only the same completed six-case evidence and write
the new `audit_v2_binding.json` and `raw_artifact_audit_v2.json` leaves. The binding
records this document, the correction tests and V2 auditor plus the exact study
and original failed-audit identities before replay. Existing files must not be
overwritten. Preserve any new failure; do not restart the physical collection.

Even a successful corrected audit establishes stationary marker acquisition
evidence only, not live maze discovery, remembered return or hardware performance.
