# N5 full-panel V8 independent review (2026-07-13)

## Verdict

**PASS.** Reviewer: `/root/camera_v8_independent_review`.

The frozen V8 implementation satisfies the isolated-verifier amendment. This
review authorizes exactly one fresh V8 infrastructure-replacement attempt. It
does not authorize a retry, scientific retry, later ladder rung, G2, held-out
evaluation, navigation, runtime, hardware, production, or promotion.

Implementation author `/root/camera_v5_independent` is different from this
reviewer. Exact execution, experiment RGB/data, production checkpoints, and
accelerators were not opened or used during review. The canonical V8 output
root remained absent.

## Frozen closure

| Artifact | SHA-256 |
| --- | --- |
| V8 amendment | `9d89acd880849e688480eddddfc8b3129570b0f0fffa7bbf54dc457ade2ec211` |
| V8 policy | `99a2777d3ba2ad8baf62b98944f05aa1affb2e74834f337a2ba0644e9c03c84c` |
| V8 executor | `f163aaf04722bb118796912bcfcdf1f4e24b7e54990e41a9d164acc08b233500` |
| V8 synthetic lifecycle | `4d11b499d4cc2ffe4a31d0ed5df73a84649947bfd8a78522556719f8af21316c` |
| V8 author tests | `700092f5ea2885e23dba03b65c5a24737060c20e934413af1886ff454ec3e5b4` |
| V8 implementation handoff | `536f31de0d8fe0cec26417b73e29ff3ef396086b05d5b2f104e7202f98df25b1` |
| Independent V8 QA | `14ba6f544e6583011cfaceb0ff3b29aac8ac615045408a3fa49f066d706c94d8` |

All frozen author artifacts were regular, singly linked files. The policy and
executor hashes bound in the authority-bearing JSON are exactly the frozen
hashes above.

## V7 terminal evidence

Static authority preflight rehashed the complete V7 source/review closure and
its terminal attempt. The attempt inventory was exactly `reservation.json` and
`failed.json`; `checkpoint.pt`, `result.json`, and `completed.json` were absent,
and both metric and gate directories were empty.

| Receipt | File SHA-256 | Content SHA-256 | Bytes |
| --- | --- | --- | ---: |
| `reservation.json` | `de5972f40743cf960d3a2b2745087504deb0a7e69c0c8ff4f557269e6563a661` | `05fd35e99471659cf507d0985f3a7e82f276456a570b135bc7b51cf8ebfc8334` | 6292 |
| `failed.json` | `fec22c763a0be6ab4796aeea88e6d9d7a59d428f2801ab1d8f827536cf5a7957` | `546a31907b63efdfdf08ed7e793ba14bba1cf6e922f2daaba24c26f21e0a0b94` | 1143 |

The failure remains stage `verification`, class `runtime`, code
`execution_failure`; journal integrity is `intact`, retry is false, and no V7
numeric payload survived or was inspected.

## Process-boundary findings

The verifier is invoked only as:

```text
sys.executable -I -B <same reviewed V8 executor> --verification-child
```

The parent sends one bounded, canonical, self-hashed request over stdin. It
binds a fresh 256-bit nonce, source-review file/content hashes, reviewed policy
and executor hashes, exact reservation/result/checkpoint/completion paths and
file hashes, parent/executable/executor identity, isolated mode, and the exact
environment contract.

The child independently checks its PPID, isolation flags, sanitized Python and
device environment, static authority, source review, source bytes, artifact
bundle, checkpoint state manifest, selected inputs, R9700 GPU0 resource, and
frozen metric computation. It calls the retained verifier's validation and
receipt computation functions directly once. It never calls the retained
publishing entry point; its compatibility writer always raises and no canonical
writer exists in the child path.

The parent rejects timeout, signal/nonzero status, stderr, empty/malformed/
noncanonical/extra/oversized stdout, request/response/source/artifact/process/
environment mismatch, publication claims, and invalid metric receipts. It has
no in-process fallback. Only after its own full receipt and artifact validation
does it journal the canonical metric receipt, then run retained finalization.

Failure remains inside the inherited descriptor-relative transaction. Exact
owned partials are removed, changed or foreign bytes are preserved invalid, a
terminal failure is written when possible, descriptors close, and retry remains
false.

## Inter-op lifecycle reproduction

A fresh CPU-only subprocess performed checkpoint-like Torch serialization,
deserialization, tensor contiguity, and hashing before calling the frozen
`configure_determinism`. The first call succeeded with one inter-op thread. A
second call in that process reproduced V7's exact `RuntimeError: cannot set
number of interop threads ...` failure. This verifies both the V7 diagnosis and
the V8 remedy: the fresh verifier child makes the frozen verifier's sole setter
call before numerical inference.

## Verification evidence

All runs hid HIP, CUDA, ROCr, and HSA devices, capped native math libraries at
one thread, and disabled external pytest plugins.

```text
V8 author/adversarial suite:                 129 passed
Retained V6 suite:                            40 passed
Applicable retained V1-V5 closure:           103 passed, 8 deselected
Independent V8 QA:                             4 passed
Isolated CPU contract smoke:                  PASS
Static authority and V7 terminal preflight:   PASS
Frozen one-call/second-call reproduction:     PASS
py_compile (V8 policy/executor/tests/QA):      PASS
Forbidden refresh/cleanup/interop scan:       PASS
git diff --check:                             PASS
Canonical V8 output root before review:       ABSENT
Exact execution, experiment data, GPU use:    NONE
```

The eight legacy deselections are only historical assertions that now-consumed
V1/V5 output or review state must be absent. No applicable science, schedule,
security, transaction, cleanup, or numerical assertion was deselected.

The CPU smoke reproduced 400 updates, 2,000 frame exposures, full five-frame
panels, schedule SHA-256
`62efec890e572623ab6d76e8c67337ee29badaf81638943ae56ed8da0a3a8634`,
and synthetic total loss `0.265` without training or data access.

## Residual limits

This is source/lifecycle authorization, not a numerical result. The exact GPU
run can still fail terminally for a numerical, resource, driver, or filesystem
reason, and the amendment permits no retry.

The supported outer command is the ordinary executor invocation, which always
relaunches itself with `-I -B`, sanitized selectors, and one-thread caps. A
caller that manually pre-isolates the outer process bypasses that relaunch, but
does not bypass the retained trainer's checks for `HIP_VISIBLE_DEVICES=0`, no
HSA override, exactly one visible AMD Radeon AI PRO R9700, and one-thread caps.
Such a manual launch is unsupported and may consume the sole attempt if it
violates the environment contract.

The child PID is reported by the fixed reviewed child over its private stdout
pipe and bound into the self-hashed response; the parent independently binds
the parent PID, executable, executor, flags, nonce, and environment. Because
the implementation uses `subprocess.run`, the parent does not compare the
reported child PID to a retained OS process handle. The fixed command, reviewed
child source, private pipe, child PPID check, and nonce make this non-blocking
for the frozen threat model, but it remains a narrower assurance than a
`Popen.pid` comparison.

## Authority record

The authority-bearing JSON is generated from
`policy.expected_source_review_core` with reviewer
`/root/camera_v8_independent_review` and exactly the frozen policy/executor
bindings. It is published after this report and all other review artifacts.
