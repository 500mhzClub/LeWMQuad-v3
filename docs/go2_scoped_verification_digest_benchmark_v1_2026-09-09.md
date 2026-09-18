# Scoped digest verification benchmark V1

Purpose: reduce repeated ancestry hashing in future development launchers while
retaining original verification conditions and exact SHA-256 artifact identity.
No running or frozen verifier is changed or replaced by this experiment.

Fixed input: completed tracking native maze 1 result
d6774bae22cb9effeb0cd85ae255de203de1539541f701d57788b58ab00769de,
its exact original launch, complete artifact bindings and original native
verify_inputs function. Admit the completed collection/raw-audit/physical-prefix
metadata and directly authenticate every bound native artifact and source before
benchmark creation. Model, dataset and scene execution are absent.

Run one fixed pair: scoped verification first, original verification second.
Both execute the same target artifact checks and original native ancestry
conditions on the same unchanged input context. Preserve each completed stage
separately. Both must finish before a complete result. One pair in this order
does not establish a controlled performance estimate; record measured wall times
and the scoped digest counters without extrapolating to controller timing.

The scoped helper clones only explicitly referenced verification functions,
retaining their code, closures, defaults, other dependencies and checks. Imports
and original module globals remain unchanged. The original digest function runs
on the first request for each path. Reuse is limited to this one call and only
while canonical nonsymlink ordinary-file metadata remains identical: device,
inode, mode, ownership, size, nanosecond modification/change time and link count.
Protected path components are rejected before content access. Every cached file
is freshly streaming-SHA-256 hashed at the end; every hash and file identity
must still match. Recheck the entire population's metadata after final hashing.
Any change, original error or failed final comparison rejects the entire call.
Clear all entries on success and failure. No persistent cache or skipped final
content check. Maximum 200,000 cached paths and 4 MiB final-hash read chunks.
Unintercepted helpers retain their original behavior and may still rehash files.

The helper does not claim an atomic filesystem snapshot or immunity to hostile
filesystem/kernel manipulation. It requires file stability throughout the
verification scope and adds identity checks around hashing; there remains an
unavoidable interval between verification and later use, as in the predecessor.
This benchmark does not itself install the helper in any launcher.

One CPU worker, 8 GiB memory admission, 128 MiB output allowance, standing
40 GiB artifact reserve. Measure CPU topology/affinity/load, RAM, GPU/VRAM,
storage and competing jobs before launch. Bounded CPU analyses may overlap
within measured headroom; no native scene is created. Use exclusive output
go2_scoped_verification_digest_benchmark_v1_attempt_001. Preserve failures;
no automatic retry, in-place resume, deletion, sealed access, source export,
training, real-robot motion or deployment claim. Verify source and direct input
bindings again and bind every output at completion.
