# Counterfactual maze dataset development V1

Preregistered collection before execution. This is a development learning corpus,
not final held-out navigation evaluation. Generate 24 independent seeded 4x4
maze graphs: 16 training and 8 validation layouts. Reject duplicate graph topology
under rotation/reflection before any execution; no runtime resampling. Width and
spawn variation are independently sampled from the declared seed. The source
cell is a leaf; its neighboring junction has two or three outgoing ports. A
randomized spanning tree plus sparse extra connections supplies the remaining
layout. All branches/frames/descendants of a layout retain that layout's role.

Collect five branches per layout, 120 planned attempts. Each reconstructs a fresh
physical scene with the same layout seed, corrected 20/0.5 gains and fixed gait.
Record 1.5 s settling, then execute the unchanged baseline prefix into the first
junction (maximum 85 command ticks). A sustained contact-free prefix crossing
permits branching even if an instantaneous arrival proxy misses. Otherwise retain
the prefix failure in coverage accounting; it produces no branch-training row.
Do not search for a more favorable prefix or retry failed starts.

From the actual prefix endpoint execute 40 command ticks (4 s) of one fixed tape:
stop [0,0,0], forward [0.3,0,0], forward-left [0.2,0,0.5], forward-right
[0.2,0,-0.5], reverse [-0.2,0,0]. Then execute five zero-command release ticks
unless already physically stopped. Absolute/slew limits and actuator latency
remain unchanged. All commands are body vx, vy, yaw rate. No learned model is
used during collection and no post-outcome action adjustment is allowed.

Stop immediately on the existing disallowed-contact/body-stability criteria,
retaining the terminating sample. Infrastructure or measurement-integrity error
stops the complete study without retries. Preserve all partial results.
Compare complete raw prefix state/command traces and the branch-start RGB/history
packet across the five independent replays; a mismatch is an integrity failure,
not additional stochastic training data. No legacy snapshot is opened.

Capture the existing causal native RGB/ideal body history at each command boundary
and terminal state. Policy observations retain their strict schema. Prefix
controller decisions, topology, camera world frames, global pose, contacts and
future outcomes remain separate audit/label artifacts. Known candidate command
tapes are valid prospective action inputs; future measured commands/sensors are
not substituted for them.

Provide outcome labels at 0.5-s increments through 4.0 s: relative x/y/yaw in the
branch-start body frame and cumulative contact incidence. If physical stopping
prevents a future sample, that motion/image target is invalid, not filled with
the terminal state. Once a contact is observed, contact-by-later-horizon is known
true as a failure endpoint; it does not imply unobserved later physics was run.
A non-contact early stop leaves later contact labels unknown. Retain terminal
reason and release motion separately.

Audit source/gait/gain identity, graph and role separation, all raw contacts,
prefix equality, candidate tape/slew, observation causality, censoring and labels.
Report branch availability and outcome diversity across layouts/actions. This
corpus is not evidence of JEPA utility by itself. Subsequent fixed-budget direct
versus JEPA experiments must keep data/sensors/actions matched, separate predictive
training from online rollout, and report executed-outcome accuracy and candidate
regret alongside latent error. Any oracle local intent used in conditional
ranking must be shared across methods and labeled as such, not claimed as an
explorer-produced goal. Final unseen-maze and physical-platform tests remain open.
