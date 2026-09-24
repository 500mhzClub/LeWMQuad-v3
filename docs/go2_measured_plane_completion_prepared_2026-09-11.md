# Measured-plane completion and next controller comparison

The running observer diagnostic is bound to launch SHA-256
`8f09edbb77d103e3fe37e6f021da16be810a1696b588dea2264e98489f30afe1`.
Its result remains unproven until the original owner ends and its complete
artifacts are verified. The observer diagnostic source and launch are unchanged.

`scripts/verify_go2_measured_plane_observer_history_v1.py` accepts the actual
result SHA-256 and creates an exclusive completion receipt. It authenticates
the original launch and ended owner, all source/input bindings, and the complete
output stream. It reconstructs each consumed public raw packet, checks the
recorded original visual/floor evidence, validates available candidate pose and
floor compositions, and reconstructs the full report and progress records.
The JSON identity adapter restores only the required episode tuple types,
including nested transported-floor anchors; it changes no numeric evidence.

This verifier does not rerun descriptor matching or point fitting, refit floor
pixels, or admit the original native execution. Its receipt states those limits.
It rejects output after the first declared stop and never treats an abbreviated
positive history as completion. Scientific candidate failures remain explicit
negative results. The checker passed **15 tests in 3.19 s**, including actual
serialized synthetic visual/floor evidence, tampering, wrong clocks, changed
scope, negative outcome accounting and live-owner rejection.

`scripts/await_go2_measured_plane_observer_completion_v1.py` may now be started
once. It checks the original owner every 30 seconds and invokes that exact
checker only after the owner ends. An original execution failure is retained;
neither the replay nor verification is automatically retried. The waiter uses
its own exclusive artifact root and records the actual result identity before
verification. Do not run the checker separately while this waiter is live.

The independent future controller comparison is defined in
`scripts/measured_plane_controller_prefix_comparison_development.py`.
It requires complete original decision reproduction, the exact authenticated
candidate observer/floor history, identical same-model raw forecasts and
correction metadata, and commands derived from actual selected actions.
It stops at the first changed requested command or terminal result, before
reading the resulting unexecuted observation. Its **13 synthetic contract tests
passed in 1.96 s**. No trained-model controller comparison or new native run has
been launched from this preparation.

Bounded live status is available through:
`python -B -m scripts.status_go2_measured_plane_and_native_v1` using the existing
development environment. It authenticates the fixed launch and exact process
identity and reads only complete bounded progress lines. File presence in that
status is not completion or navigation qualification.
