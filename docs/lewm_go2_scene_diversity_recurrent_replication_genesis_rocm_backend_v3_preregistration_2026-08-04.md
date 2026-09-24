# Genesis ROCm backend V3 host-identity successor preregistration

Date: 2026-08-04

Status: source-only preregistration. This document creates no qualification,
scientific, GPU, retry, resume, refill, deployment, or navigation authority.

## Question and material delta

V2 was consumed before Genesis initialization. Its `env -i` launch passed the
declared Python, selector, `ld.lld`, and `rocminfo` gates, but the isolated
identity subprocess aborted while importing Genesis because native
`get_repo_dir` requires `getenv("HOME")` and V2's claimed complete environment
did not bind `HOME`. A controlled read-only replay reproduced SIGABRT without
`HOME` and passed the exact identity query with
`HOME=/home/andrewknowles`.

V3 tests one hypothesis only:

> Explicitly binding, sanitizing, validating, child-overwriting, and
> receipt-auditing the literal non-secret host identity
> `HOME=/home/andrewknowles` removes the pre-initialization abort.

Aside from fresh V3 identities, roots, role-local caches, witnesses, and the
single `HOME` entry, the canonical V2-to-V3 plan delta must be empty. In
particular V3 retains V2's lexical ROCm venv launcher, exact unresolved
`ld.lld` Unix-driver entrypoint, resolved-target negative control, selector
sanitation, replay semantics, 64-scene scientific panel, data, model, seeds,
caps, evaluation, physical gates, and two-scene qualification order `[12, 0]`.

`HOME` is a source literal. It must never be derived from the ambient process.
Before any qualification or scientific reservation, the outer interpreter and
the complete role environment must match exactly. Every child environment
must remove ambient `HOME` and then write the literal expected value. The
identity receipt must contain `home=/home/andrewknowles`, and replay validation
must reject a missing or mutated value. `USER`, `LOGNAME`, and `LANG` remain
absent rather than becoming additional identity inputs.

## Fresh custody

Scientific identity:
`go2-scene-diversity-recurrent-replication-genesis-rocm-backend-v3`.

Qualification identity:
`go2-scene-diversity-recurrent-replication-genesis-rocm-backend-v3-qualification`.

Both use fresh V3 attempt roots, collection roots, and role-local Quadrants
caches. Neither V2 root is an input or output.

The sole admissible V2 failure witness is:

- `docs/lewm_go2_scene_diversity_recurrent_replication_genesis_rocm_backend_v2_qualification_terminal_review_2026-08-04.json`
- SHA-256 `166aec87b6e61d62116069a12472b768c3ff462c09cf1e6088af62ab7397dd0e`
- 16,198 bytes

The V2 terminal, reservation, authority, cache, worker, scene, RGB, process,
result, replay, and other runtime payloads are forbidden. V2 source files may
be bound only as source witnesses. V2 exact plans are not V3 source-closure
inputs. Qualification output is never reusable by science.

## Qualification and decision rule

Before authority can be considered, focused source tests must establish:

1. Canonical V2-to-V3 plan comparison shows only fresh identity/witness fields,
   role-local cache paths, and exact `HOME`.
2. The actual lexical child interpreter remains the V2-reviewed ROCm venv.
3. A read-only import oracle reproduces missing-`HOME` SIGABRT/134 without
   Genesis initialization and passes with the exact literal identity.
4. Missing, ambient-derived, or mutated `HOME`, and ambient `USER`, `LOGNAME`,
   or `LANG`, fail before delegation or reservation.
5. The qualification identity receipt and scientific replay validator both
   require exact `home` evidence.
6. The V3 source closure binds only reviewed source inputs and the V2 terminal
   review document; all fresh roots and authorities remain absent.

Only a separately reviewed one-shot qualification authority may consume the
V3 qualification root. Qualification retains V2's `[12, 0]` probes, watchdog,
timing extrapolation, HIP/R9700 identity, EGL, kernel-reset, contact-force,
physics, and process-evidence gates. Only an exact qualification PASS may make
a separately reviewed scientific authority eligible. Scientific gates and
decision thresholds are unchanged.

## Stop rule

If V3 reveals another missing ambient host variable before any scene begins,
do not create a V4 that adds one more scalar environment variable. That would
be evidence that the bootstrap contract was not understood. The next step
must instead be a comprehensive, independently reviewed bootstrap dependency
audit covering native-library host identity, filesystem, locale, loader,
cache, and process assumptions before any further successor is proposed.
