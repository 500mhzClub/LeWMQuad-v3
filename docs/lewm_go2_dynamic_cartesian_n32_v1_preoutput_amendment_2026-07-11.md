# Go2 Dynamic-Cartesian N32 V1 Pre-Output Amendment

Date: 2026-07-11

Status: frozen before N32 smoke output, authoritative model output, holdout
access, or result publication.

This amendment resolves three execution ambiguities in
`lewm_go2_dynamic_cartesian_n32_v1_binding_2026-07-11.md`. It changes no
dataset, panel, role, model intervention, optimizer, budget, metric, gate, seed,
or held-out comparison.

## Finalizer Output

The two seed result files named in the binding remain the only authoritative
N32 artifacts:

```text
.generated/go2_dynamic_cartesian_n32/v1/seed_20260710_result.json
.generated/go2_dynamic_cartesian_n32/v1/seed_20260711_result.json
```

The torch-free finalizer is validation-only. It reads an exact result path and
externally supplied file SHA-256, recomputes every pure contract, schedule,
gate, access-ledger, source, and publication invariant, prints a canonical
summary to standard output, and creates no file. The runner owns private
staging and immutable no-replace publication of each seed result. Therefore no
third aggregate artifact is introduced.

## Two-Seed Order

Both registered seeds run exactly once, in the order `20260710` then
`20260711`, regardless of whether the first seed is favorable. The second
runner invocation must bind and validate the immutable first result and its
externally supplied file SHA-256 before any model or panel artifact access.
Neither seed may select, replace, or trigger a retry of the other.

## Inert Model Weight

`occupancy_weight` is frozen to `2.0`, matching the immutable static patch-7
comparator configuration. N32 trains through `occupancy_logits` plus the
direct equal-capacity hierarchical loss, so this stored model field is not
multiplied into the N32 objective. The runner and finalizer must nevertheless
bind and validate the value as part of the exact model configuration.

## Memory Execution

N32 uses the unchunked dynamic projection and attention path at the bound batch
size of four frames on discrete GPU0. Query chunking is not an N32
intervention. It may be added and separately parity-tested before later shared
training, where the larger paired batch creates the relevant memory pressure.
The Raphael integrated GPU remains forbidden.
