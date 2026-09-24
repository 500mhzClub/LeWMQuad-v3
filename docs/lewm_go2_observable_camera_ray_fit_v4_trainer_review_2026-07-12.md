# Observable camera-ray fit V4 trainer source review

Date: 2026-07-12

Status: **independent PASS recorded; final narrow binding awaits different-agent byte review**

## Reviewed boundary

The independent review verified the all-false authorization snapshot and its
36-file transitive source closure. The bound snapshot file/content/source-map
SHA-256 values were:

- `f31eabe61e18c4c1087487bee4a3dbb9f84b2f76a62782af08fcd5a7910e53a6`;
- `ae3e5b70b48ec664eae5494f877754554a7696672ce589f2faced9eae49670e9`;
- `d7d9c4220ddf4f3c4e54b1f04a5d15b531a722f88cad0f47b51cecadaea717ec`.

At review time, the trainer, launcher, ladder gate, and ladder finalizer file
SHA-256 values were respectively:

- `c0d7efba9d5c59be4900da40a9694de2689f27ff04c27f635c99d3644873e6a6`;
- `b71fe0e3552aa9a2d66121422f4d31503a929160ea356d135032dc4e16e7dd9c`;
- `68de1d6394df0c0bfc5c0a6fa8963f4f6f3cab5d57ca24139077b80c82a1086c`;
- `598caa65896c7a553769f170eba3f9410681e305be8a76d4d994fd588b39970c`.

## Blocking findings

1. A caller could construct a self-consistent N=1 stage mapping containing
   nonexistent result/completion hashes and fabricated metrics; the mapping
   passed and authorized N=4 because execution validation did not reopen the
   canonical attempt artifacts. The seed validator had the same defect.
2. Metric validation recomputed ratios only from supplied aggregates. It did
   not reload the checkpoint and exact selected inputs, rerun inference, or
   independently reproduce losses, confusion matrices, depth values,
   quantiles, raster NLL, and family decisions. Arbitrary finite checkpoint
   tensors plus invented perfect metrics could therefore pass.
3. The finalizer opened reservation, result, checkpoint, and completion bytes
   before checking canonical paths and authorization.
4. The serialized launcher receipt was forgeable by same-user, same-process
   code once authorization became true. Its nonce and bindings were not an
   external authority.
5. Reviewed source bytes were hashed before a later process execution/import,
   leaving a replacement and bytecode-cache interval between review and use.

The review did **not** reproduce a reported duplicate `@property`; the bound
trainer contains one decorator and direct access succeeds.

## Required successor

The successor must constrain paths and authorization before any restricted
open; execute the trainer only inside the verified launcher from already-hashed
source bytes; reject direct trainer invocation and serialized capabilities;
reopen and reverify every canonical predecessor artifact; and use a separately
licensed verifier to rerun checkpoint inference over the exact selected
targets/RGB assignments and independently reconstruct every gating number.
Seed promotion must repeat the complete four-rung chain.

No V4 RGB, model output, checkpoint, G2, runtime, held-out, or sealed access is
licensed by this review. CPU-only synthetic remediation and a different
independent source review are required before N=5 can run.

## Unreviewed remediation candidate

The candidate successor now uses a one-process content-addressed source loader,
rejects direct trainer execution and serialized environment receipts, verifies
loaded module origins and source bytes after import, constrains finalizer paths
and both authorizations before artifact bytes or Torch, recursively reopens and
byte-recomputes canonical stage/seed chains, and requires a separately licensed
full-inference metric receipt. Synthetic regressions cover direct invocation,
same-PID serialized receipts, malicious bytecode caches, concurrent source
replacement, pre-canonical opens, dummy 33/44 gates, mapping-only artifacts,
invented metric receipts, and four-stage seed revalidation.

This is not a review decision. The metric-verifier authorization remains all
false, exact per-rung target-partition signatures are now frozen, the trainer
authorization is regenerated all-false, and a different independent
reviewer must evaluate the new closure before any execution can be licensed.

The V2 amendment replaces N1/N4 with N5 and freezes the four target
partitions. The gate, trainer, launcher, metric verifier, and finalizer now bind
the exact freeze file/content hashes, verifier source hash, and amendment hash.
The metric verifier reproduces all 180 target files and the ordered target-byte
commitments before checkpoint import or inference. This closes the previous
pending-target-constant finding, but it is not a review decision.

The regenerated 42-file candidate snapshot remains all false. Its current
authorization file/content/source-map SHA-256 values are
`373fa9e89c631f6a99ae3e71fb7d1fd0a464f9223dcc3259997ac661a7131250`,
`4eb212826b5bbc7b6b4fae77d297d53ff971e809e4cf2d4027a67778315c1964`,
and `7977d69706a5a12d682e2bb5d9d53cf7e46c048f32b10581808298c939c47dad`.
No PASS is asserted here.

## 2026-07-13 Second Independent Review

Verdict: **BLOCK; no V4 fit execution authorized**.

The review confirmed the corrected `(5,16,32,320)` partitions, schedules,
thresholds, target-byte reproduction, full-inference metric reconstruction,
and all-false authorizations. It nevertheless reproduced two high-severity
authority defects without opening protected payloads:

1. Public `create_verified_launch_context(receipt)` accepted a mapping with
   only the expected schema, then minted the private token/live identity. With
   authorization and review still pending, that forged context reached
   `load_verified_trainer` and imported the trainer and Torch.
2. The metric verifier rehashed disk sources but used ordinary imports for the
   gate, launcher, trainer/model, and finalizer. A preloaded or temporarily
   substituted module could therefore remain live after the reviewed disk file
   was restored, preserving the prior hash-to-import gap.

The focused review suite produced 168 passes and three stale builder/auditor
fixture failures unrelated to these blockers. The remediation candidate now
uses the active `N=5/16/32/320` four-stage ladder wording and records why those
three obsolete, upstream-bound fixtures cannot be rewritten in this closure.
It also performs canonical preflight inside the context factory and uses
captured-byte, fresh-identity loading throughout metric verification and
finalization. These are candidate changes, not a PASS; a new independent
review remains mandatory.

## Remediation Candidate After Second Review

The schema-only public receipt minting path and the entire context/token/live
registry API are removed. Following the governing
`lewm_go2_scientific_execution_authority_threat_model_2026-07-13.md`, each
canonical CLI now performs fixed-path authorization/review/source preflight and
immediately invokes a captured private entrypoint in the same one-shot process.
It returns no context, token, issuer, or loader. With the current pending review
and all-false policy, the launcher fails before trainer, NumPy, or Torch import.

Metric verification and finalization now reject relevant canonical modules
already present in `sys.modules`, capture and rehash the complete bound runtime
source graph, compile it under a fresh content-addressed module namespace, and
check module identity, logical source name, origin, and captured source hash
after load. Canonical wrappers only preflight and dispatch; receipt computation,
publication, artifact/gate validation, checkpoint validation, target-partition
reproduction, and inference execute exclusively in captured private modules.
No source file is replaced or restored during loading.

CPU-only synthetic verification produced 176 passes with the three previously
reported builder/auditor fixtures deselected. Those fixtures are objectively
stale, but their bytes are part of the already-reviewed upstream builder/auditor
manifest and completed dataset provenance; changing them here would invalidate
that independent upstream closure. They remain documented rather than being
silently rewritten. The fresh captured runtime graph, including trainer and
Torch with GPU visibility disabled, also loaded successfully. This section is
a remediation record, not an independent review or execution authorization.

## 2026-07-13 Fixed-Graph Candidate Review

Verdict: **BLOCK; no V4 fit execution authorized**.

The latest candidate removed caller-provided graph mappings and mutable
`__name__` authorization. Its launcher/trainer hashes are
`787ce8422c7c83d14b5bfa1210e0b35396da6f6d2cd8bb5ac654b591c77db36b` /
`dc51a47f06ab18399d765804700e74cdc285680d162fa350d57f39453d198496`,
and its all-false authorization file/source-map hashes are
`921afa9bfe2597c1311d812f490bf1666f005788cfd2a6637d8fc711b4beb1b4` /
`dc7b9245a2d1d8896d54113f31fbe31e0060eead11e3022c6d76f543cb50e7d5`.
The focused launcher/trainer/finalizer/verifier suite passed `60/60`; the full
V4 collection produced `181` passes and the same three frozen upstream fixture
failures.

The original capability-return defect nevertheless remains. Under the actual
Torch interpreter, an isolated caller imported the launcher by fixed path,
constructed `_ContentAddressedRuntime` with the canonical pending authorization
hash, received the live runtime object, loaded
`scripts.train_go2_observable_camera_ray_fit_v4`, and imported Torch while all
trainer and metric licenses remained false. The caller can no longer choose the
source graph, but preauthorization still returns an execution-capable object.
This contradicts the one-shot/no-capability-return threat model and the prior
remediation claim above.

The successor must make runtime construction lexically internal to each
reviewed one-shot operation, perform full canonical authorization before that
construction, and return only the operation's terminal result. Launcher,
metric verifier, finalizer, and RGB workers must not expose a loader, runtime,
finder, source mapping, or caller callback. A permanent regression must run
the isolated fixed-path reproduction and prove that neither trainer nor Torch
is imported while authorization is pending.

## Post-Fixed-Graph Remediation Candidate

This candidate removes module-accessible content-addressed loader, finder, and
runtime classes; source-capture/load/reverification functions; runtime-module
maps and proxies; and injected RGB-worker callbacks. Runtime source capture,
loader/finder construction, and module loading now remain lexical to the fixed
launcher, metric-verifier, finalizer, and RGB-worker terminal operations. Each
terminal performs the complete canonical authorization preflight before that
machinery is constructed and returns only its terminal operation result.

The permanent regression uses the actual
`/home/andrewknowles/TinyQuadJEPA/bin/python -I` interpreter, imports the fixed
launcher path, invokes the canonical pending authorization, and observes
`fixed-pending-no-runtime-trainer-torch`. It also asserts that the removed
capabilities are absent and that neither the captured trainer, a synthetic V4
runtime namespace, nor Torch enters `sys.modules`. The CPU-only focused suite
passes `55/55`. The full V4 collection passes `176` tests and retains the same
three frozen upstream builder/auditor fixture failures already documented
above. No dataset payload, training, inference, or GPU was used.

The regenerated 42-file candidate remains pending and all trainer and metric
authority flags remain false. The fresh authorization file/content/source-map
SHA-256 values are
`38b58b8f119347d520f16761cad56ead80bc2be9e4293a8b40f62c296d537d47`,
`21cae4a1eb986e103de2a47ec24d5c650fb48b0213e7c781abfe43a1cde42ca1`,
and `0cf65c798edde164de273ca7f609a9f89eba51cbf5422b01583f4afbc7efa027`.
The launcher, trainer, metric-verifier, and finalizer source SHA-256 values are
`a9ca7f572fc6cd327198011f62bfb309ce11bb91eee1aa3c637379817dc36185`,
`0248759e79125c0c0dfa46807989807f090f665c12be5efdd50dd7924254028d`,
`c07bc01cdd70379a4829da752cc38888252070f625d83eb41d07d5b69318ec2b`,
and `84f25389e0ea64c82b96bb82fd2c9e305ddfe2f8b2b110eb015358e4b0965733`.
This is a remediation candidate only. It records no PASS and authorizes no V4
data access, training, checkpoint use, runtime use, G2, holdout, or promotion.

## 2026-07-13 Post-Fixed-Graph Independent Review

Reviewer: `/root/v4_final_independent_review`  
Decision: **PASS; no remaining source blocker**

The independent reviewer reconstructed all 42 sorted, unique, exact-role source
entries and reviewed the all-false post-fixed-graph closure without editing
files or opening restricted payloads. The reviewed authorization
file/content/source-map SHA-256 values were
`38b58b8f119347d520f16761cad56ead80bc2be9e4293a8b40f62c296d537d47`,
`21cae4a1eb986e103de2a47ec24d5c650fb48b0213e7c781abfe43a1cde42ca1`,
and `0cf65c798edde164de273ca7f609a9f89eba51cbf5422b01583f4afbc7efa027`.
The reviewed pending review-record file/content SHA-256 values were
`db23289f1b9cad5d1ea7d5d448c2068b5d6db36f26ef6b3b1445ad54ad9849e3`
and `ef8ba81a755eee06fa52148c75877f5faf0c0c9bf14452d2be8ee545ae31f0b0`.

The reviewed launcher, trainer, metric verifier, and finalizer SHA-256 values
were
`a9ca7f572fc6cd327198011f62bfb309ce11bb91eee1aa3c637379817dc36185`,
`0248759e79125c0c0dfa46807989807f090f665c12be5efdd50dd7924254028d`,
`c07bc01cdd70379a4829da752cc38888252070f625d83eb41d07d5b69318ec2b`,
and `84f25389e0ea64c82b96bb82fd2c9e305ddfe2f8b2b110eb015358e4b0965733`.
Independent evidence was `55/55` focused tests in 1.05 seconds, `176` full-V4
passes plus the same three documented frozen upstream fixture failures in 7.78
seconds, and exact TinyQuad `python -I` output
`fixed-pending-no-runtime-trainer-torch`. No data payload or GPU was used.

## Final Narrow Execution Binding Candidate

The deterministic stdlib-only binding tool enabled exactly two trainer fields:
`development_fit` and `development_checkpoint_creation_authorized`.
`checkpoint_use_authorized`, G2, runtime, holdout, and promotion remain false;
aggregation eligibility remains false throughout the trainer/checkpoint/gate
contracts. The separate metric policy enabled only verification-only checkpoint
use, selected-train target access, selected-train RGB access, model inference,
and metric-receipt creation. Its G2, runtime, holdout, and promotion licenses
remain false, and the policy remains non-authoritative.

The verifier now hard-binds the final metric policy. The review record and
trainer authorization bind the recomputed 42-file closure and the already
frozen dataset/audit identities. Final SHA-256 values are:

- metric authorization file/content:
  `091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf` /
  `c4090f47b417d5766f5d5100615b2f1c3891a8340e2813ad089bf894beeb98d2`;
- metric verifier source:
  `d106a85ea08c2335d0816316c970b31cfaf9842874dd98c6924abaa8077d9b89`;
- review record file/content:
  `6009d44dd0ae9ce55627728c1c157f40671eb07112144231c0ef170e31120aa0` /
  `429cb4a936ff9186bf8463ef3970493266cf40d31ce24c36d16a529b114ca339`;
- final source map:
  `40d9c7dff078d3942e19d2047f37015e5052b866fef681fb7a624540fa1f3ed6`;
- trainer authorization file/content:
  `c3fe277898f8247b630c554735e3b3ee6663dda78fd0b57f480cfe44b1ac4729` /
  `4c14f514ec784208c75b7f6c5c0779e7cbd55818cd53ed47fd09a2eb27904f80`;
- deterministic binding tool:
  `e81562a869a2abcf85e88417cb91f67ac27ca288ec78dee2c50c8a55f371f632`.

CPU-only metadata/fail-closed verification passed `55/55` focused tests and
`19/19` ladder-gate tests. No RGB, target, checkpoint, result, metric-receipt,
training, inference, or GPU operation occurred. The ladder has not been
launched. Because the metric policy, verifier constants, metadata-facing tests,
review record, and final source map changed after the independent all-false
PASS, a different agent must byte-review this exact final binding before any
execution.

## 2026-07-13 N5 Provenance Preflight Incident

The first exact N5 invocation stopped during frozen dataset-manifest metadata
validation. The trainer constant expected
`beddb29b9826d7a21968effea863c040a6cfc9849ab0b2a78c4105d28dbb37d2`,
while the frozen builder, reviewed implementation manifest, and completed
dataset manifest all bind
`beddb29b9826d7a21968effea863d040a6cfc9849ab0b2a78c4105d28dbb37d2`.
This was a one-byte source transcription defect, not a dataset mutation.

The failure occurred before audit or target-shard payload loading, attempt
reservation, RGB access, GPU validation, model construction, inference, or
training. The canonical development-fit output root and N5 attempt directory
were absent after the failure, so the reviewed one-attempt budget was not
consumed.

The remediation corrects only that trainer constant and adds a focused
manifest-only preflight regression which makes reservation, dataset-payload,
RGB, GPU, and training functions fail if reached. The deterministic binder may
recompute only the trainer and its focused test entries from the exact prior
final binding. Trainer and metric license dictionaries are unchanged. The
resulting candidate requires a different-agent byte review before N5 is
launched.

The regenerated candidate identities are:

- corrected trainer source:
  `70255e4bece10af7a1736887614e24d6cf1bdd6cc8da5c40cdf74570b2ea21d3`;
- focused trainer regression:
  `ff64aa071661a40e5f2dc1118cce60755d3a4dc7286c0a40dd0bd1fe82a42f1f`;
- 42-entry source map:
  `084509f97ef6dc95a24877ff3205b26b88bad9595dca3f168cf76376655cd2f1`;
- review record file/content:
  `61fee8fbc4a356ca772af9dc41213ce4ad8a1426ef8059f9f9e1223f29e8c8c6` /
  `b533610e3f5ca9e8831392f1c6ce85a0d666e18edd2ae5e234ca5855d74f3684`;
- trainer authorization file/content:
  `11a9a4ea6274d5c02194a8ec6de4465ede00c699ec1f4fbf792cf0ebb0354255` /
  `3655916d33f561d91a48c7c884537e31884a2782e6f58e3f3df76a9b9fd59810`;
- deterministic binding tool:
  `3ffcf7fa7e3bec2491e5fff0b14d0809bb0b80c7580070a491a518ec37d373cc`.

The metric authorization file/content and verifier source remain exactly
`091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf` /
`c4090f47b417d5766f5d5100615b2f1c3891a8340e2813ad089bf894beeb98d2`
and
`d106a85ea08c2335d0816316c970b31cfaf9842874dd98c6924abaa8077d9b89`.
The focused CPU suite passed `56/56`, the ladder-gate suite passed `19/19`,
and the isolated manifest-only regression passed. The production manifest
validator returned the exact `...863d040...` commitment while the development
output root and reservation remained absent. The binder then reran twice with
identical output and no file changes.

## 2026-07-13 Ladder-V3 Failure Successor Candidate

This section supersedes only the earlier statement that N5 remained
unreserved. After the corrected provenance preflight was independently
reviewed, V1 N5 was launched and consumed its one reservation. It terminated
before publication because PyTorch appended
` (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)` to the
otherwise byte-exact allowlisted `grid_sampler_2d_backward_cuda` warning. The
old matcher rejected that trailer.

V1 is now immutable. Its reservation file/content SHA-256 values are
`115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e` /
`ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd`;
its failure file/content values are
`6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48` /
`7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4`.
The V1 attempt inventory is exactly `reservation.json` and `failed.json`; no
checkpoint, result, completion, gate, or metric receipt exists.

The ladder-v3 successor writes only to `development_fit_v2`, uses V2 schemas,
and binds the immutable V1 failure in every new reservation, result, and
checkpoint metadata chain. The original rungs, seeds, steps, data, target
partitions, model, thresholds, and licenses are unchanged. The warning parser
accepts only one optional exact `/pytorch/aten/src/ATen/Context.cpp:<positive
ASCII decimal>` trailer after an otherwise byte-exact allowlisted warning. It
records raw and normalized text and the parsed line, while changed kernels,
text, paths, punctuation, nonpositive or leading-zero lines, and duplicate
trailers remain fatal.

Candidate identities are:

- frozen amendment:
  `86718d072fe151b9419318c204d4130147e098150d4fd80557f9d5865dc8f9f3`;
- 43-entry source map:
  `eb8c97dae6f3ef3839a886cac200774c87dfb6e452f71c13e75557eb8c9feac3`;
- review record file/content:
  `c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea` /
  `ab55270986268c5a326eeb6ba191cd9a0531112b1b742812d2cbd549f67158be`;
- trainer authorization file/content:
  `d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802` /
  `18a285e80252d41de7daadba918a00223d8770b71c533f74807e0ace5444ac1e`;
- deterministic binding tool:
  `aab3d862c6ad59abf8d446d7c505e500fc6f59514570b49b84321c9a25c19cba`.

The metric authorization remains byte-identical; its verifier source is now
`235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f`
because it is confined to the V2 root and V2 schemas. The focused capped-CPU
suite passes `85/85`. The broader V4 collection passes `146` tests and retains
the same three documented frozen upstream builder/auditor fixture failures.
No V2 directory was created, no V2 N5 reservation was made, and no protected
payload, V2 training, checkpoint inference, or GPU operation occurred. This
exact candidate requires different-agent byte review before V2 N5 execution.
