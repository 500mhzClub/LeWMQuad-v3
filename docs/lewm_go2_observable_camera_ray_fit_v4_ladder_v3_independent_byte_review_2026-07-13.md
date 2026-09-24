# Observable camera-ray V4 ladder-v3 independent byte review

Date: 2026-07-13

Status: **PASS for one V2 `N=5`, seed `20260710` development attempt only**

## Scope

This review is independent of the ladder-v3 successor author. It reviewed the
exact frozen successor bytes after the terminal V1 warning-parser failure. It
does not authorize G2, held-out data, runtime use, promotion, threshold changes,
checkpoint reuse, a second attempt, or any rung beyond the immediate fail-fast
successor step.

The immutable V1 attempt remains exactly two files:

- reservation file/content SHA-256:
  `115e3a4e0ad7db7f5bd6b01c7ddde29d79563600ffb84ef77a0c585f009e854e` /
  `ca458f9371a211017f1b7a710b41508e2219a1afe19516ace2553a8eaa4d15dd`;
- failure file/content SHA-256:
  `6eb1becc195165e5fb49c1d222cac301f4169f301a48245d23a2b8213363af48` /
  `7c1fe8f1ea73d8caef33debd9076bc3ddcacfaf337ec2a0000cec64f678c21e4`.

There is no V1 checkpoint, result, completion, metric receipt, or gate. The new
`development_fit_v2` root was absent throughout this review.

## Findings

1. The successor changes the output root and artifact schemas, binds every new
   reservation to the exact terminal V1 evidence, and preserves one attempt per
   seed/rung. V1 cannot be reused as a V2 output namespace.
2. Rungs, seeds, step counts, batch sizes, optimizer, learning rate, weight
   decay, data, target partitions, model, thresholds, wrong-RGB control, GPU0
   policy, and license fields are unchanged.
3. Warning normalization accepts only the two existing byte-exact warning
   bodies, optionally followed by one exact positive-ASCII-decimal
   `/pytorch/aten/src/ATen/Context.cpp:<line>` trailer. Raw, normalized, line,
   trailer, kernel-inventory, and kernel-count evidence is retained and
   independently reconstructed by the result gate.
4. Changed kernel, body, punctuation, path, filename, zero/leading-zero or
   non-decimal line, duplicate trailer, and arbitrary suffix variants reject.
5. The bound authorization contains 43 source entries. Every file digest, its
   canonical source-map digest, and the authorization content digest were
   independently recomputed and match.

## Exact reviewed identities

- amendment: `86718d072fe151b9419318c204d4130147e098150d4fd80557f9d5865dc8f9f3`;
- source map: `eb8c97dae6f3ef3839a886cac200774c87dfb6e452f71c13e75557eb8c9feac3`;
- bound authorization file/content:
  `d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802` /
  `18a285e80252d41de7daadba918a00223d8770b71c533f74807e0ace5444ac1e`;
- review record file/content:
  `c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea` /
  `ab55270986268c5a326eeb6ba191cd9a0531112b1b742812d2cbd549f67158be`;
- gate: `aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad`;
- launcher: `71d95ae79cd90c64bee8b06f2787b336d72e2fca1e23fcb0cc52f921350a2ff4`;
- trainer: `299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa`;
- metric verifier: `235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f`;
- finalizer: `375b1dcd3a548cf7b130fb67291ef5116effcc0197a28be42643bfc59e710ec6`;
- metric authorization: `091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf`.

## Verification

- focused successor tests: `85 passed`;
- broader V4 closure: `187 passed`, with only the same three frozen upstream
  fixture failures already recorded before this successor;
- exact authorization content/source-map/file reconstruction: PASS;
- immutable V1 inventory and file hashes: PASS;
- bytecode compilation: PASS.

The license is void if any identity above changes. The `N=5` result must be
verified and finalized immediately after publication. A structural failure or
numeric gate failure stops the ladder; a pass licenses only the next frozen
rung through the existing sequential gate.
