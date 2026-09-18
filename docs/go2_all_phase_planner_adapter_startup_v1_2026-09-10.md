# Expanded-model planner interface adapter: raw startup comparison V1

All six original expanded-model maze2 workers failed at observation3 with
`training-only translation wrapper in evaluation mode required`. Their complete
raw audits reproduced that failure. The selector's isinstance check accepts
TrainingTranslationBiasModel, whereas the admitted expanded correction loader
returns the separate AllPhaseTranslationBiasModel. Both are evaluation-only;
the failure occurs before invoking the expanded model, so these runs provide
no learned-navigation comparison. Preserve the six original failures and their
parent/waiter completion. Do not retry or mutate either running original owner.

The separately named AllPhasePlannerModel implements the accepted correction
interface by subclassing TrainingTranslationBiasModel while copying the already
admitted expanded model's base and buffers. It preserves all state keys and
tensor bytes, owns independent copies, and uses the exact existing expanded
forward method. Expanded correction receipts keep their original schema and
full original admission. No training, fitting, old-data admission substitution,
checkpoint selection, global monkeypatch, or source mutation is performed.

The focused tests invoke the actual existing selector with full/no-RGB and
single/two-head synthetic models, check exact complete forward outputs and
selected forecasts, preserve clock/padding semantics, reject invalid modes and
gradients, and verify independent tensor ownership. Initial padding testing
correctly rejected an invalid nonzero padded fixture; the fixture was corrected
and invalid padding is also explicitly tested as a rejection case.

Replay the first four actual public observations for every completed original
worker, with two fresh copies of the same assigned state and the unchanged
ResidualAnchoredContinuationController. Reproduce every complete original
decision, preserve the three exact warmup decisions, public packets, observed
map and contact history, then require the adapted model to reach real planning
at observation3. Compare every raw expanded forward output and the planner's
forecast bank exactly. The original recorded decision contains no forecast at
that frame; equality is against direct execution of the original expanded
wrapper on the same causal inputs, not a nonexistent saved forecast.

Stop at the first terminal difference at observation3. Do not read observation4
on the changed controller trajectory or execute its requested command. Native
pose/topology is not used by the replay controller. The complete original worker
artifacts and launch are hashed before/after; original top-level completion can
still be pending during this read-only replay. The old source/input admission
is retained and its model loader rechecks its bound artifacts; the full nested
input verifiers are not rerun by this diagnostic.

The earlier reached-frontier implementation remains separate and unexecuted.
This adapter isolates the model-interface failure with unchanged navigation
logic. A later prospective physical comparison must use a newly named reviewed
launcher, preserve the old six-case result, and reproduce the 900-sample,
four-observation startup before comparing new physical behavior. Constructor
checks and equal state hashes alone are insufficient; this replay requires
actual inference through the full controller.
