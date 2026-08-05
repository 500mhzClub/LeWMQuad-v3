# Shared JEPA V5 raw-supervision Builder V6 independent review

Date: 2026-07-13

Reviewer: `/root`

Verdict: **BLOCK**

## Frozen candidate

| Role | SHA-256 |
| --- | --- |
| Builder source | `88c36063e257d9d163317abb15d7854f3da783e0ec15537da4c3d62b113740d7` |
| Builder CLI | `089aca4882f4f574be7972914c12c05acabf1cd898bea6f59422bf07b94f828d` |
| Builder test | `acf5ca8cdd829d1c3c4ef44dbc4fe7e5d2f05a7dc7ec01662b60d9f27ececdd0` |
| Author handoff | `d2cf130a9e2c902776327f6bd71a1b1f363a4dcfde6df0e2aba15edc3957e80b` |
| Independent QA | `2c74e3315be3443bab11a3b7896df4df29d8b233b634b7ab539123386bc0c89a` |

The implementation author is `/root/raw_builder_arch`, distinct from this
reviewer. The V6 amendment and frozen predecessor identities reproduce.

## Decisive finding

Builder V6 closes the V5 source-versus-staging race with one retained-FD and
inotify-backed transaction, but its final transaction boundary does not close
the retained publication ancestry.

After the owned no-replace rename, `refresh_after_owned_mutation()` validates
the complete publication chain and `validate_after_rename()` validates the
dataset inventory. The builder then fsyncs the publication parent and calls
`require_final_quiet()`. That last operation only drains inotify events; it
does not call `retained.validate()` or repeat the bound inventory checks.
Ancestors above the watched publication directory are retained but not watched.

The independent behavioral test performs the following sequence:

1. complete the exact owned rename and post-rename inventory validation;
2. fsync the retained publication parent;
3. move the canonical ancestor containing that parent and replace the original
   ancestor path with an empty directory tree; and
4. call `require_final_quiet()`.

V6 returns successfully. The retained dataset exists below the moved ancestor,
while the canonical dataset path is absent. A final complete retained-path
match was therefore not proven when the transaction closed. This violates the
amendment requirement that watches/descriptors close only after the post-rename
queue is quiescent and all retained paths, hashes, and fingerprints still
match.

## Evidence

CPU-capped independent QA result:

```text
1 passed, 1 failed

FAILED
test_final_quiet_rejects_publication_ancestor_move_and_replacement
Failed: DID NOT RAISE RawSupervisionBuildError
```

The passing test binds all four frozen author artifacts. The failure is the
decisive exploit above. No exact authorization, protected payload, canonical
dataset, accelerator, G2, held-out, runtime, hardware, or production input was
opened.

## Required successor

Builder V6 is terminally ineligible for authorization. A pre-implementation
V7 amendment must bind this BLOCK and require the final quiet operation to
revalidate the complete retained publication chain and bound source/staging
inventories after its last event drain and parent fsync. A mutation between
that validation and descriptor close must be excluded by one final closed
operation or fail closed. Auditor V6 may remain an implementation input but
cannot receive exact authority through the blocked Builder V6 map.

The canonical BLOCK JSON has content SHA-256
`c639170b672180c8943e08efaff8d23063e8773488d1ff0f77beeb4ce44dd74b`
and file SHA-256
`55d50a38f0c7d23e4ff537b124db3b9f24a24ea5b30413ff6be1ac381870c163`.
