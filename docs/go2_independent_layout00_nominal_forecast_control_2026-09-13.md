# Nominal forecast control on independent layout 0

The existing `nominal` case is queued after the JEPA, reactive,
supervised-rollout and direct cases. It uses the same layout 0 and existing
`seed_2026091001_full_jepa` assignment. This addition was made while the JEPA
case was collecting, after inspecting partial development decisions, and before
any completed independent-maze outcome. It is a development comparison.

The intervention replaces learned candidate forecasts with nominal
requested-twist forecasts. The existing cost, candidate action interface,
observed residual correction, perception, map, mission, 8,000-decision budget
and physical evaluator remain. The assigned JEPA model is loaded for the shared
interface, but nominal selection does not call its forward method. No model is
trained or selected by this addition.

Compare this case primarily with the original full-RGB JEPA case to assess
learned forecasts versus this nominal forecasting assumption. Nominal forecasts
still predict ahead; this does **not** isolate prediction on/off or remove
persistent memory. The reactive case remains a whole-method comparison.
All five cases use **one independent maze**, not five independent replicates.

Queue PID 3152516, creation time 1789263046.07, tool session 24389 waits for the
existing direct queue owner, PID 3149894, creation time 1789261729.43. It uses the
unchanged one-case runner after operational completion, including negative
scientific outcomes. There is no automatic retry. Native collection and audit
remain serial, and the runner checks hardware at actual launch. The exact
assignment and owner receipt is `go2_independent_layout00_nominal_queue_2026-09-13.json`.

Output root: `go2_stop_conditioned_independent_00_nominal_seed_2026091001_full_jepa_v1_attempt_001`
under the existing RecoveryStorage navigation development artifact base.
Physics pauses during computation; no real-time or hardware claim follows.
