# Next workstream specification: evaluation-first place representation

This is a specification only; no implementation or training is authorized in
the current block.

## Evaluation first

Freeze a held-out place-retrieval panel before training. Verify every frame,
node, graph target, family assignment, and target identity. Run the complete
evaluator on random descriptors, same-frame identity descriptors, native LeWM,
mean-pooled ViT-L, and an untrained place head. Emit the complete result
schema before training. Independently report graph coverage so descriptor
quality cannot be confused with absent destinations.

## Single seed

Only after evaluator fixtures and untrained-model execution succeed, train one
place model and evaluate it completely. Stop if weak. Replicate only after a
positive go/no-go decision under `EVALUATION_FIRST_SINGLE_SEED`.

## Architectural boundary

Keep three interfaces distinct: predictive dynamics representation,
viewpoint-invariant place/memory representation, and safety/utility
representation. Do not assume one latent serves all three functions. No
adapter, place head, scorer, graph modification, or model is implemented here.
