# V4 metric-verifier prepublication failure

Date: 2026-07-13

Status: **terminal verifier-source finding; N5 training artifact preserved**

## Event

The first `development_fit_v2` `N=5`, seed `20260710` training attempt
completed and atomically published its reservation, checkpoint, result, and
completion artifacts. The separately licensed metric verifier then failed
before creating a metric-verification directory or receipt with:

`PermissionError: V4 spawned RGB terminal differs from captured source`

The rejection occurred in the captured trainer's `decode_selected_rgb` source
boundary. Training launches from the canonical launcher, so the trainer finds
that launcher's captured multiprocessing terminal. Metric verification launches
from the canonical verifier; the unchanged trainer nevertheless requires the
live `__main__` path and code identity to be the trainer launcher. It therefore
rejects the legitimate captured verifier before RGB decoding completes.

## Immutable N5 evidence

- reservation file/content:
  `f5926ee9006df8d163a2d1a17882d82124608ddce319ea0fb5e80fcfe2c2a8aa` /
  `699b4e95ed05cb13a79fe6af8507fae5d987af9ff1977b0e4684f32742aa4943`;
- checkpoint file/content:
  `f1739c742f9c19d5e17753da504a547254eb6e1997bb1ac4eca8b188bbf1dcf0` /
  `589060417903167bbf9ce7605c906b25cd802edd73b79ec607c77403c6df305a`;
- result file/content:
  `39030bb7928a6b078b03156dc9e14fb206c60c73ab2acac88bfd307c5a65bbfa` /
  `8c38e13f411a5cd9b03362cb5ac98379875065f284a75ac894706944ff252b61`;
- completion file/content:
  `4fb9b5629f039ac16692ec6e171a8188f3bf8b7d052ac8cde26b8ac86c10f6af` /
  `48022dca829a73b7cbd3b665ac7679807825a9aefd56a48e752ae07e6eaa336f`.

The attempt inventory is exactly `reservation.json`, `checkpoint.pt`,
`result.json`, and `completed.json`. It has no failure artifact because the
training attempt itself succeeded. There is no metric-verification receipt,
stage gate, seed gate, larger rung, G2 artifact, held-out access, runtime use,
or promotion.

## Source identities at failure

- trainer: `299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa`;
- launcher: `71d95ae79cd90c64bee8b06f2787b336d72e2fca1e23fcb0cc52f921350a2ff4`;
- metric verifier: `235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f`;
- bound trainer authorization:
  `d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802`;
- metric authorization:
  `091d26f6be0372c003528be370028e6f431bcdef9770ce3855d8b1cf4045a3cf`.

## Successor rule

Training must not be rerun and the N5 artifacts must not be edited, moved, or
re-published. A successor may only reopen these exact bytes under a new,
different-agent-reviewed verifier authority. It must bind the old training
source map and this incident, preserve the same target/RGB/checkpoint/metric
scope and GPU0 policy, derive RGB from exact selected files, and publish at
most one receipt with exclusive writes. It must not weaken source isolation by
accepting a caller-provided decoder or metric. A failed structural or numeric
verification stops the ladder.
