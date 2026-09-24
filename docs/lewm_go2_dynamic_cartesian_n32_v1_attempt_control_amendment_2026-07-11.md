# Dynamic-Cartesian N32 v1 attempt-control amendment

Date: 2026-07-11  
Status: frozen before any N32 model output or G2 payload access

This amendment closes the crash window left by the execution binding and the
pre-output amendment. It changes no model, data, optimizer, metric, gate, seed,
or result identity.

## Authoritative attempt markers

Each registered authoritative seed has one canonical immutable control marker:

- seed `20260710`: `.generated/go2_dynamic_cartesian_n32/v1/seed_20260710_attempt.json`
- seed `20260711`: `.generated/go2_dynamic_cartesian_n32/v1/seed_20260711_attempt.json`

The runner must create the seed's marker atomically with no-replace semantics
after validating the implementation manifest and all bound source hashes. For
seed `20260711`, it must also validate the complete seed-`20260710` result and
its attempt marker first. Marker publication must occur before any panel,
sidecar, image, label, model state, model construction, or model-output access.

The marker binds the seed, invocation, UTC start time, canonical result path,
execution binding, both amendments, implementation-manifest path plus external
file and canonical-content hashes, and (for seed `20260711`) the exact primary
result and primary attempt-marker identities. The marker records that the
authoritative attempt is consumed, retry is forbidden, and payload access had
not started at publication. Its canonical content hash and external file hash
must be bound into the result.

An existing marker is an unconditional hard failure, even when the result file
does not exist. A crash after marker publication consumes the seed permanently;
the marker must never be removed, replaced, repaired, or treated as permission
to retry. Non-authoritative smoke runs must neither create nor accept an
authoritative attempt marker.

## Validation and finalization

Result validation requires the externally read marker object and file SHA-256,
and must verify the marker/result invocation, timestamp, seed, manifest,
amendment, primary-evidence, and canonical-path bindings exactly. Seed-pair
validation requires both markers and both externally observed marker file
hashes.

The torch-free finalizer is validation-only. It reads the canonical marker with
no-symlink regular-file checks, requires an externally supplied marker file
SHA-256, validates it against the result, rechecks it for drift, and creates no
artifact. It never creates, changes, or removes a marker.

## Resource policy

Attempt control is CPU-only metadata work. It does not alter the frozen R9700
GPU0 policy, six-worker CPU hashing cap, one-native-thread-per-worker cap, or
the prohibition on using the Raphael integrated GPU.
