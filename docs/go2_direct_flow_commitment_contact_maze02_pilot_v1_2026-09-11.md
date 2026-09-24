# Contact-plus-flow maze 02 physics pilot V1

Run one fresh development episode only after the original tracking, extended
budget and sustained-turn native experiments have completed their existing
queue. Admit their complete original inputs and preserve all failures through
the frozen contact-plus-flow input module. This pilot changes only visual
tracking inside the original contact-scoring controller.

Use original layout 2, its original scene, robot, sensor and renderer settings,
the original full supervised-rollout model and correction admission, and the
original 3000-tick mission budget. Use a fresh controller and memory, one spawned
scene worker, deterministic model execution and one thread for BLAS and OpenCV.
Keep physics paused during computation and report measured wall time honestly.
No sustained-turn or longer-budget policy change is adopted here.

The completed raw replay is bound to launch
`6c577445f7bff2e58aad960c6d26908c683a20b0d7a4fbb2344967b5072da9ad`
and result
`a4404e737a29f2499ee5313fcb116bc7a980f6fff5c2f75122ba6be6df437560`.
It preserves all 561 earlier decisions and 558 earlier forecast comparisons,
then recovers the failed observation at frame 561 and requests a left turn.
That replay has not executed the changed command or observed its consequences.

The fresh episode must match all 28,800 physical samples before that command,
all 562 public sensor packets and complete prospective controller decisions.
Require the actual boundary command and all 50 of its physical samples.
Do not require later physics to match the failed original trajectory. A raw
controller replay from actual collected sensors, unchanged model-state checks,
renderer witnesses, complete command audit and the original strict physical
visibility and native round-trip criteria remain mandatory.

Preserve complete unsuccessful episodes and integrity failures. Require
exclusive output creation, resource assessment before launch, monitoring during
execution, all artifact bindings and final input revalidation. No automatic
retry or resume. Source preflight may validate code and resources while the
queue is active, but must not admit runtime inputs or create an output scene.

This is one reused development maze. It does not establish independent-maze
generalization, JEPA benefit, memory benefit, real-time execution or hardware
qualification. No independent-study policy is selected by this pilot.
