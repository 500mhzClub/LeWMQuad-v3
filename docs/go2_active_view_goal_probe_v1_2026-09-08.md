# Active-view learned waypoint native probe V1

Test actual view acquisition followed by onward navigation on the two fixed
known integration layouts 052 and 039. Keep the final family full-JEPA snapshot
`bcb8874e2adf89053463206267a4ccb90380909c324e303734a59b038f5b1821`, original
1.2-m mission goal, six candidate plans, five-tick commitment, 240 navigation
ticks, arrival/quiet test, ten terminal-zero intervals, corner observer and
native physical/measurement guards. No training, threshold search or retry.

Every accepted frame updates the frozen joint-visual floor map and persistent
surface memory. At each replanning boundary obtain the frozen floor-waypoint
proposal. When a route exists, select the first route-cell centre at least
350 mm from the current position, or the final centre if none meets that distance.
Use that intermediate target in the existing learned terminal-distance/contact
utility. The original mission goal alone controls arrival. An unobserved initial
connector remains explicitly unobserved; this native exploratory controller
does not claim that the proposal is a certified floor or footprint route.

When no route exists, acquire another view with only hold/left-turn/right-turn
candidates. Choose the first scan direction once from the larger count of retained
floor squares to either side within 2 m in the current body frame (left wins ties).
Use fixed absolute map headings in order: signed 45 degrees, signed 90 degrees,
opposite 45 degrees, opposite 90 degrees, 180 degrees. Advance only when measured
heading is within 0.1 rad of the current target and no route exists. Exhaustion
latches a terminal view-budget failure. A route may interrupt scanning.

For scanning, score each learned half-second outcome by
`0.4 * wrapped-heading-error reduction - predicted XY drift - 1.2 * contact score`.
Use the same model predictions, with complete original forecasts and scores
retained. The robot does not rotate from a fixed command tape; learned forecasts,
actual measured heading and newly acquired map evidence drive every replan.
For either phase apply the frozen half-second persistent surface-intersection
filter, then maximize phase utility with original bank tie order. No admissible
phase candidate means terminal zero drain, not success. Do not waive conflicts
to reach a view target or extrapolate a missing pose.

Admission requires the complete floor replay
`1c482b9a875092c9866d8085832426e95c802f60ca094bc4b52c4261428f1182`, its frozen
source witnesses, all six-fit/model bindings, robot URDF and inherited native/input
identities. Freeze new source and tests before launching exclusively at
`go2_active_view_goal_probe_v1_attempt_001`. Run `active_view_family_episode_052`,
then `active_view_family_episode_039`, each with a fresh process/scene/model/
observer/map. Check hardware and storage before launch; run one worker serially
for uncontended full-loop timing, with at least 32 GiB available RAM and 8 GiB
planned output above the existing 40 GiB reserve. Monitor actual resources.

Audit all sensor/map/model/command replay, state invariance, contacts, goals,
visibility, all map/view failures and complete-loop timing. Scientific failure
does not cancel the second fixed case; infrastructure/raw-audit failure does.
Retain phase transitions and whether any waypoint command actually ran. Successful
scanning alone is not goal-reaching. This is not an independent-maze, realistic-
latency, hardware or physical-backtracking qualification. Physics remains paused
during computation; the known timing overhead is reported rather than hidden.
