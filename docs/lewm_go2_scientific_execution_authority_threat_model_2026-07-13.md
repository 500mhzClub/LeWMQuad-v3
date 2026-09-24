# Go2 scientific execution-authority threat model

Date: 2026-07-13

Status: **governing clarification for V4-V8 execution evidence**

## Objective

Execution authority exists to prove which reviewed code and role-bound inputs
produced a scientific result. It must prevent accidental privilege leakage,
self-reported metrics, stale or substituted artifacts, repeated selection, and
controller access to evaluator or hidden geometry. It is not a cryptographic
security boundary against arbitrary code already running as the same operating-
system user.

Python module globals, underscore names, object identity, closure cells, random
tokens, and `id()` registries are not secrets from arbitrary in-process code.
Treating them as such creates false assurance: reflective code can inspect or
mutate them, and same-user code can normally read the underlying files anyway.
No future gate may claim protection solely from a Python capability token.

## Protected failure classes

Authoritative execution must reject:

- caller-authored aggregate metrics, pass flags, access ledgers, counts, or
  hash-shaped producer identities in place of raw outcomes;
- wrong, copied, alternate, symlinked, or path-escaped repository/role roots;
- source changed between review, import, inference, and finalization;
- already-loaded canonical project modules or dynamic plugin/user-site imports;
- wrong checkpoint/model state, role manifest, scene ordering, calibration,
  threshold, target bytes, seed, predecessor, or attempt reservation;
- direct controller access to evaluator feedback, hidden scene geometry,
  held-out/sealed roles, or a second selection attempt;
- serialization/copy/replay being mistaken for live evidence issuance;
- a test/synthetic outcome becoming production-eligible.

## Execution model

1. Each authoritative stage is a one-shot CLI process. It performs fixed-path,
   no-follow canonical preflight and immediately executes the captured,
   rehashed source graph. It returns no context, token, loader, issuer, or
   production-capable helper to callers.
2. Start with user site and dynamic project imports disabled. Project modules
   execute only from captured reviewed bytes under fresh private identities;
   the canonical wrappers perform preflight and delegate, but do not recompute
   scientific outputs themselves.
3. The runner reads raw role inputs and emits immutable per-instance outcomes
   and actual-open events. Test-only issuers are separate and irreversibly
   marked production-ineligible.
4. A separately reviewed finalizer reopens the exact raw outcomes and derives
   every count/metric. No caller mapping can substitute for model inference,
   geometry evaluation, or execution outcomes.
5. Fixed attempt registries and role identities enforce one-shot use. Paths and
   namespace components are closed vocabularies or exact SHA-256 values.
6. Controller code is itself part of the captured reviewed source graph. It is
   not an arbitrary plugin environment and receives only the registered runtime
   observations/memory, never evaluator callbacks or hidden-role file handles.

## Stronger isolation

If arbitrary third-party Python must run beside the controller, same-process
source review is insufficient. That configuration requires an operating-system
boundary such as a separate uid/container, restricted file descriptors,
seccomp/sandbox policy, and an external outcome broker. Until implemented,
arbitrary same-user reflective code is forbidden inside authoritative
processes and must not be simulated with secret Python objects.

## Review rule

Reviews must still call public and module-level helpers directly to ensure none
can create production evidence from caller data. Closure introspection is a
valid finding when a returned/importable capability is part of the design. The
remediation is to remove the capability-returning API and use the one-shot
process model, not to hide the token in another Python object.
