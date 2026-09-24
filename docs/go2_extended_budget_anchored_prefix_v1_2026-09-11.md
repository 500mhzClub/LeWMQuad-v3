# Actual-execution prefix comparison for the budget follow-up

The longer-budget source protocol's reference to a budget receipt field is
made precise here: two scalar paths differ, `shared_navigation_budget_ticks`
and `mission_receipt.global_navigation_ticks`. The original sixth-case first
four recorded decisions and the original controller/mission source contain
both. They must have exact integer values 3000 in the original and 4000 in the
candidate. No other field, including a selector's view-budget state, is removed
or normalized. The frozen predecessor preparation is left unchanged.

The comparator processes observations 0 through 3003, with full normalized
decision equality expected only before observation 3003. At that boundary the
original must report budget exhaustion and request zero. The candidate's actual
terminal and command are reported, not assumed successful. Earlier decision,
request, command-tape or public-packet drift is preserved as a negative finding.
Neither more time nor a matching prefix establishes return success.

Physical equality covers the first 150,900 samples, through acquisition of
observation 3003, before executing its command. Exactly 3,003 completed command
intervals precede that observation. Full tape rows and decision requests must
agree. Every paired public packet through the boundary is reconstructed and
fingerprinted. No later public packet or decision is compared; replay readers
load complete manifests and history arrays as their existing implementation
requires. Later physical outcomes may differ and are not equated.

The actual-file comparator requires explicit bindings for every consumed raw
input, including the last paired camera files, and rechecks supplied bindings
before and after. Its caller must authenticate the launch, assigned model,
source closure and full raw audit independently. It grants no such admission
itself and does not replace visibility, sensor, model, physical or success
audits. It is source preparation with synthetic tests, not a new experiment.
