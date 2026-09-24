# Measured recovery-publication lock repair

The instantaneous arm of the interrupted repaired cohort stalled after 1565
camera frames. Its main thread waited for the controller request lock, while
all threads were waiting and simulated time stopped advancing. Source inspection
and a focused old-versus-fixed test expose a circular dependency: the visual
recovery publisher holds that lock while waiting for measured simulation time;
the simulator needs the lock to request a command before advancing time.
Live worker Python stacks could not be obtained. Preserve that distinction
between the reproduced mechanism and direct observations of the native stall.

`MeasuredRecoveryPublicationMixin` waits for measured service completion outside
the controller lock. Inside the lock it takes a non-waiting clock snapshot and
performs the original cancellation/publication. Duplicate triggers still do
nothing. Regression tests verify that the original method prevents simulator
lock acquisition, the replacement allows it, and the recorded publication time
includes advancement while waiting to acquire the lock. Three tests passed.

Run one separately labeled technical validation on the exposed first maze with
the instantaneous arm and the same frozen supervised model. Keep all navigation,
sensor, timing, candidate-count and clearance settings fixed. Report whether
recovery publication was exercised, whether the bounded mission terminated
without deadlock, and the physically evaluated navigation outcome. A tracking
failure or budget exhaustion remains a navigation failure even if execution
terminates correctly. This is not an independent-maze result, a replacement for
the interrupted attempt or completion of the broader goal.

Reuse the already measured sequential CPU setup (0–7,16–23), with no competing
simulation and no heavy concurrent analysis. About 4.9 GiB artifact space remains
after retiring the interrupted cohort's diagnosed first-success depth; its two
failures and older full repair references remain. Preserve this new attempt and
do not repeat it until a favorable outcome appears.

Launcher: `scripts/run_go2_measured_recovery_publication_development.py`, first
`--prepare`, then no arguments to run, then `--evaluate` after owner exit.
The interrupted 20-run cohort is closed at four outcomes. Any later matched
comparison uses a new fixed plan; the other three generated mazes have not yet
been executed by any arm.

The single assigned run completed and passed physical goal-and-home checks in
229.22 simulated seconds, with zero contacts and 536/550 plans on time. Both
quiet arrivals passed and the return reversed all seven observed outbound
corridor edges, with no invalid transitions. There were no pipeline faults.

The corrected recovery publisher was exercised twice, at camera frames 1892
and 1923, cancelling two older command windows. No old nonzero command was
requested strictly after either publication, and there were no same-clock old
command ties. The run terminated normally. This gives live exercised evidence
alongside the three deterministic regression tests. It is not replay of the
stalled trajectory and does not prove that this change alone explains the
different navigation outcome. The interrupted result remains unchanged.

The complete technical/navigation readout is `recovery_publication_fix_readout_v1.json`
under `go2_measured_recovery_publication_instantaneous_noise_2mm_native_layout00_4800_v1_attempt_001`.
Retain this exercised repair recording in full as the current publication-fix
reference. Use the corrected publisher consistently across every arm of the
next prospective comparison on the three unused layouts. Keep that new cohort
separate from both the interrupted four attempts and this exposed-maze run.
