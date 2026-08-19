# PLACE_HEAD_DEV_V2 development result

Source HEAD at launch: `e082cc0a42f39e35210613081a15f969c4f9bc9a`.

The frozen manifest was split scene-disjointly with seed `2026081901`: 52 fit
scenes, 18 evaluation scenes, 2,336 fit rows, and 864 evaluation rows across
the available six evaluation families. The inventory contained 4,907 unique
RGB frames; all resolved.

The ViT-L token cache contains 4,907 `[768,1024]` FP16 entries (7,718,671,744
bytes). Cache-index SHA-256:
`e00366bba42bcd21c358156a8bc9c861d66860090b7baa104d6baaee7887f007`.

One PLACE_HEAD_DEV_V2 seed (`2026081901`) was trained for 30 epochs with the
registered cosine-margin triplet loss. Loss decreased from `0.10828853` to
`0.00111243`. Final checkpoint SHA-256:
`9d9dbc93f2b8f4d0c4b470944d244a7197ba235e7e37507bd7594d5eab0e6ddb`.

Held-out node retrieval (864 queries; same-scene gallery) was:

| condition | top-1 | top-3 | MRR | median rank | mean margin |
|---|---:|---:|---:|---:|---:|
| trained head | 0.011574 | 0.263889 | 0.202642 | 7 | 0.000947 |
| mean-pooled ViT-L | 0.010417 | 0.306713 | 0.214792 | 6 | 0.000098 |
| untrained head | 0.010417 | 0.283565 | 0.206936 | 7 | 0.000057 |

The trained head does not meet the prespecified top-1 `>=0.50`, improvement
`>=0.15`, or family-robustness thresholds. Development classification:
`PLACE_HEAD_DEVELOPMENT_NO_SIGNAL`.

No graph-aware metric, predictor evaluation, Genesis run, or planning screen
was performed. Exactly one place-head seed was trained; no predictor was
opened. The external cache and checkpoint remain under the designated cache
root; no processes remain running.
