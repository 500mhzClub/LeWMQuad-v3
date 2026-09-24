# Compact raw-depth archive helper prepared

The completed native storage diagnosis found that derived primary/auxiliary
depth and validity payloads account for 45.6328% of recorded bytes. The new
`scripts/compact_native_depth_archive_development.py` implements a candidate
storage format that preserves raw optical depth and auxiliary diagnostic
segmentation while omitting those derived payloads. It does not replace any
existing archive, capture method, packet builder, audit or launcher.

`write` creates a new `compact_primary_depth_FRAME.npz` or
`compact_auxiliary_depth_FRAME.npz` exclusively. Primary archives contain only
the original float32 optical-depth array. Auxiliary archives additionally retain
the complete native integer segmentation array. The helper preserves array bits,
including nonfinite raw depth and signed zero. It returns a storage binding and,
for auxiliary frames, a separate evaluator segmentation binding. The future
caller must bind these to the actual acquisition clock/calibration and the
complete trial artifact roster; the helper does not establish those identities.

`read_native` decodes only optical depth. `read_evaluator_segmentation` is a
separate evaluator-only entry point. Both authenticate the archive against its
byte count and SHA-256. The format requires exactly the declared members,
DEFLATE compression, 480-by-640 native arrays, a bounded NumPy V1 header and
exact payload lengths. It rejects object arrays, extra members, forged shapes,
trailing array bytes, changed tensor identities and symlink/protected paths.
Reads are bounded to 5 MiB before archive parsing. Member-size and header checks
precede array allocation. Integer frame indices are bounded to 0 through 8013.
Those bounds describe this candidate format, not a native execution admission.

The existing depth and auxiliary-RGB packet builders remain responsible for
calibration, range masking, freshness, public schema and RGB/depth pairing.
Neither diagnostic segmentation nor evaluator camera pose is introduced into
the public observation packet by this helper.

## Verification

Source SHA-256:
`d8da2451e9a77197b61cbd37ab795ae7cbc7fa0d582da811491fd48a6b44baee`.

Test source `lewm/tests/test_compact_native_depth_archive_development.py`,
SHA-256:
`699ee4c12c95a45a977f16fb9e5cc74d3dae040782cd87cdfb8f6b8c4c0ec44e`.

Focused tests passed on their first invocation: **38 passed in 2.10 seconds**,
session 19619, exit 0, using the original deterministic single-thread environment
and `python -B -m pytest -q -p no:cacheprovider` with that exact test path.
They use synthetic native rasters with range boundaries, nonfinite values and
a nondefault NaN bit pattern. Tests establish bitwise depth/segmentation round
trips, equality through the existing complete primary/auxiliary depth and
auxiliary RGB constructors, fresh returned arrays, exclusive preservation of
existing files, separate public/evaluator decoding, and rejection of malformed
bindings, arrays, archives and headers. The synthetic fixture's compact files
are smaller than its corresponding legacy raw-plus-derived archives. That is
not a measured saving for the complete actual native population.

Source-only verification, session 87946, exited 0 with 2,688 paths in the union
of prepared native/comparator/factory/archive source and test dependencies.
All 2,656 files bound by the live prefix replay were independently rehashed
unchanged. The short tests overlapped that non-timing prefix comparison; no
additional full controller replay or native scene ran. The prepared longer
native output remained absent.

Before adoption, the next evidence needed is a separately bound actual-recorded
archive comparison, followed by complete packet and raw-audit compatibility,
measured storage and capture/reconstruction cost, and prospective capture/audit
integration with a revised resource definition. The current prefix and the
already prepared 8,000-step native trial retain their original file formats.
No old artifact was deleted or re-encoded, no real checkpoint or camera was
used by these tests, and no actual-run storage or latency saving is claimed.
