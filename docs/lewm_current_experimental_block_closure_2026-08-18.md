# Current experimental block closure

Status: **CLOSED — TRAINED_BUT_UNEVALUATED_PANEL_UNAVAILABLE** for
`FROZEN_VITL_SELF_SUPERVISED_PLACE_HEAD_V1`.

This is a read-only synthesis of preserved receipts. No model, predictor,
held-out panel, memory graph, utility shard, or simulator was opened for this
closure. Nothing is running.

## Scientific record

The established predictive result is that RGB-plus-control-history two-step
rollout supervision improves direct counterfactual fidelity through H1–H4 and
improves selected action-specific retrieval outcomes relative to the RGB
one-step comparator.

The registered equal-family H2 interaction for deployment-valid proprioception
was `0.0006128598490125481`, with two-sided 95% t interval
`[-0.0018562554956278992, 0.0030819751936529954]`. The RGB
rollout-minus-one-step H2 effect was `0.008076707101566105`
`[0.006428472675623277, 0.009724941527508933]`, versus
`0.008689566950578653` `[0.006974169328166152, 0.010404964572991154]`
with proprioception. This supports the null that deployment-valid
proprioception does not materially amplify rollout supervision.

The four-step horizon trade-off is preserved as a diagnostic ablation. At H4,
four-step-minus-two-step equal-family effects were: changed-token cosine
`+0.014186333399265998` `[0.0110914503017043, 0.017281216496827696]`;
normalized-error reduction `+0.024730200072923647`
`[0.019280345267437424, 0.03018005487840987]`; top-1 retrieval
`+0.005859374999999983` `[-0.017336684719289452, 0.029055434719289418]`;
MRR `+0.0012957028744789295` `[-0.017033904280901357, 0.019625310029859216]`;
and pairwise discrimination `+0.003708964646464627`
`[-0.009408298650947791, 0.016826227943877045]`. Longer rollout improves
H4 fidelity but does not establish additional action differentiation.

The fixed-pooling ViT-L utility scorer is a scientific qualification failure.
The ViT-g utility scorer is an exploratory `NO_SCALING_SIGNAL` result. The
attentive-readout utility scorer is a technical non-result. The non-learned
ViT-L shadow-memory and native-LeWM retrieval assays fail their true-target
upper-bound gates. The place-head seed trained successfully but was not
scientifically evaluated because its required frozen 48-row panel was
unavailable; this is not a performance failure.

## Preserved artefact identities

The eight-seed factorial and 240-branch predictor qualification remain frozen.
The two-step final-analysis report digest is
`60b0bb2d0b13ba47eac5e306c33d97dcfdce31102870edfc50b01f7f9b247161`; its
factorial manifest digest is
`6ff053033475debd3d8bb415080efb15adfaefc31f01295b956bd85c12b6dac0`.

The four-step line is bound by repair commit
`dc94fdfd0e8d29f65643a34981f901cc7dcd5bcb`, smoke
`f0fea98f2f03bc857a7fbf99f9d8f2a4f26f6172b8da2ae57bfb3dccfa163b5c`,
resource preflight `ac6163541a9919050f688814d2ab8d67c47402de2b60c8020418d3073c56a779`,
training receipt set `cfe0f8a3398bdab6547215a7dc1a50b777b1c2aa2051ff7661d4faf728a7ebe5`,
evaluation terminal `7d754e0d84564184607bf7e0738eff285db8e47f72fa15bc662732538ef42786`
(raw SHA-256 `8b8c81b1597008a6f74fc166196f266889200a61cd63fbbd926d67d2ca330061`),
four-step contract `823a722dffc2a13843bd2c5936bd46d5bf7de4399d1323d691e7f778d12d5100`,
common H4 manifest `9857af70e482fdde16074fbacb1b9676565a1936d82de0020588162536b4dd39`,
and target-cache index `5b0de5c12efb85cb8c06dfa4ac9884fd5f0ad5f76fa151c580539f37eb5bfd02`.
The preserved evaluated successor receipt has result digest
`8e3722b58bff497e8a59efb9b255e451b0b7e25746744bc8bb6fdf3ce4b53c85` and
checkpoint-set digest `0664a4b7040395191ef614a865e6279dcab1522478f2312c5d4783730b6a056c`.

The place-head closure is bound to seed `2026081801`, checkpoint
`83613a000dbce616666a884a506a5df75f498cce7e1ce73fa82c84359240c66a`, and
training contract `f7443ccae7096c1c0277ce0e225ddb77f86060a6e02ad9c3c811c85d73d3f549`.
Its inventory was eight families, eight training scenes, 32 nodes, 64 view
records, and 286 unique frozen RGB frames. Smoke passed; loss moved from
4.1601457596 to 1.0999858379 over 30 epochs. The unavailable-panel terminal
is `PLACE_HEAD_TRUE_TARGET_NOT_REACHED_PANEL_UNAVAILABLE`.

## Compute, storage, and failed paths

The place-head optimizer phase took 4.84 seconds; its five runtime files total
approximately 2.7 MB. Existing factorial, rollout, scorer, and memory receipts
remain the accounting authority for their completed runs.

Infrastructure paths not to repeat include repeated source/API repair loops,
target-cache relocation attempts, broad scorer source-closure validation,
global ViT-L shadow-gallery retrieval, multiview/locality descriptor variants,
and native-LeWM retrieval as a proxy for a V-JEPA place space. None justifies
weakening custody or substituting a different held-out corpus.

## Claims boundary

Supported thesis/paper claims are limited to: action-conditioned rollout
supervision improves predictive counterfactual fidelity; the improvement is
visible in selected action-specific retrieval diagnostics; deployment-valid
proprioception does not materially amplify that effect; and longer rollout
training trades earlier-horizon behavior for improved H4 fidelity without a
demonstrated planning or utility gain.

Not supported: planning, utility-scoring, navigation, a viable V-JEPA-to-LeWM
interface, a viable place head, any multi-seed place-head claim, or a claim
that four-step training improves action differentiation. The place head is
trained but unevaluated.

The architecture boundary remains explicit: predictive representation
(RGB-plus-control-history rollout) and place representation (memory-compatible
descriptor) are separate interfaces. Predictive fidelity does not imply place
retrieval; require a complete frozen panel and evaluator before making that
connection.

## Prospective development rule

`EVALUATION_FIRST_SINGLE_SEED` is active. For every future model: freeze a
held-out panel; verify every identity and target; run the complete evaluator
with a fixture or untrained model; generate the complete result schema; train
exactly one seed; evaluate it completely; and replicate only after a positive
go/no-go decision. No new model is proposed or implemented in this closure.
