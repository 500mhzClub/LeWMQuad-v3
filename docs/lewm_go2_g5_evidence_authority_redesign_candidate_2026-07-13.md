# Go2 G5 evidence-authority redesign candidate

Date: 2026-07-13

Status: **PASS for posterior/issuer source; production runner and promotion remain blocked**

## Boundary change

The reversible sparse-posterior mathematics is retained, but its former G3
binding and observation producers are no longer production interfaces.
`TargetMemoryContextIssuer.bind_g3`, public context/positive/negative issue
methods, and the four module-global capability objects were removed.

The retained fixture types are explicitly named `Synthetic*`, are not exported,
and expose only `_for_tests` issue methods. Their issuer contract and memory
serialization are permanently marked:

- `synthetic_only=true`;
- `production_authority_eligible=false`.

Changing those serialized fields is rejected even when a caller recomputes the
outer content hash. Copies, deep copies, dataclass replacements, parsed
serialization clones, payload mutations, and caller-created issuer instances
do not gain synthetic issuance identity.

## Exact object-identity closure

A follow-up adversarial audit found that `object.__new__` plus
`__dict__.update` could clone a synthetic issuer while retaining its context
capabilities and issuance ledgers. It also found that a shallow or object-shell
clone of `ReversibleTargetBeliefMemory` retained the writer capability and
could read or mutate the live posterior.

The issuer now keeps its allocation owner outside `__dict__` and every
authority-bearing issuer operation checks that exact object identity. The
single-writer lease now records both the exact memory object and its capability;
matching copied fields are insufficient. Ordinary shallow/deep memory copies
are rejected, and object-shell issuer/memory clones fail before consuming a
context, evidence record, or posterior mutation. Exact regressions reproduce
both original clone attacks and verify that the real issuer and writer remain
usable afterward.

## Production status

`scripts/run_go2_g5_runner_owned_observation_v1.py` defines the future raw
runner input boundary. It has no caller arguments and excludes caller-selected
candidate domains, localized distributions, unlocalized mass, visibility
probabilities, producer identities, and LOS contracts.

All six production identities are hard-unset. The CLI therefore stops before
opening any G3 outcome, RGB frame, observation checkpoint, episode authority,
or promoted-output path. It produces no artifact.

## Verification

Different-agent review by `/root/downstream_readiness` passed the 50-test
focused posterior/issuer suite and the 351-test adjacent physical-memory/claim
suite. Independent probes reproduced the exact issuer object-shell, shallow
copy, deep-copy, and target-memory object-shell cases. Every copied object was
rejected before evidence consumption or posterior mutation, and the genuine
issuer/writer remained usable afterward. All six production identities remain
`None`, and the production CLI still fails closed before input access.

No G2, held-out, sealed, protected-data, rollout, GPU, or promoted-output
execution occurred.

- posterior implementation:
  `b7f42f90accc9b44f9c38c386318e6775a26d3184d03086d14904487384f14f3`;
- fail-closed one-shot runner:
  `f7009462fc53e7c23adfe21fe8f6cd2d40b42753ab192536097812eb26e756a8`;
- posterior tests:
  `813ede3e46770b41d617ab90efb5e43ba77c4f99e411c44ce4638f2707cc90ce`;
- authority tests:
  `b3507fb837a3dc8f983cee8290a0e288b5a8e8d05ed999a9dbc5e79a4d6f6a98`.

This candidate proves fail-closed admission and retained posterior behavior. It
does not implement or authorize the captured runner that will derive G3
domains, observation-head distributions, or physical visibility evidence.
The result is an exact Python-object ownership boundary under the governing
threat model; it is not a claim of cryptographic isolation from arbitrary
same-process reflective mutation.
