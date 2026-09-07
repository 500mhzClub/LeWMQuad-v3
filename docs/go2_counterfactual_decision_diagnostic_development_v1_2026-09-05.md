# Conditional executed-branch decision diagnostic, development V1

Secondary analysis specified while seven of nine fixed learning models had
completed, after seeing their aggregate motion/contact metrics but **before
computing any action-choice/regret results**. This is not retrospectively part
of the original learning preregistration. No fitting, checkpoint selection,
physical recollection or final-test access is involved.

Question: do the audited outcome predictions choose useful actions among the
five actually executed, same-state counterfactual branches? The scope is a
single open-loop local decision with a supplied local intent, not autonomous
exploration or closed-loop navigation.

For each of the eight development-validation layouts, evaluate three equally
weighted body-start displacement intents: forward `(0.8,0)` metres, left
`(0,0.8)` and right `(0,-0.8)`. These fixed task inputs do not reveal whether an
exit is open. All methods get identical intents, initial observations and five
candidate plans. Use only the already fixed final 4-s outcome predictions.

Predicted cost is `10 * predicted_contact_probability + Euclidean distance`
between predicted displacement and the supplied intent. There is no fitted
calibration, changed threshold, heading term, intent-dependent action restriction
or coefficient search. Candidate ties resolve by the existing fixed action
order. This scalar is a declared development utility, not a calibrated physical
safety guarantee.

Realized cost is 10 for any native-verified contact by 4 s; otherwise it is the
distance from the measured 4-s displacement to the intent. Therefore censored
post-contact motion is never imputed or scored. An unavailable noncontact 4-s
outcome invalidates the whole layout's candidate comparison, not just that
candidate. Also report the fixed all-stop choice and the best realized candidate
as explicit baselines. Regret is chosen realized cost minus best candidate cost.

Apply the rule to all three training conditions, every trained inference head,
all three seeds, the original simple baselines, and the existing fixed shuffles.
Report each intent/layout/seed choice, contact frequency, stop frequency,
candidate agreement with oracle minimum, realized cost and regret. Reduce three
intents within layout first and seeds within layout second. Do not use 24
layout-intent pairs or 72 seed-intent pairs as independent maze replications.
Keep the learned comparison's descriptive layout-bootstrap convention where
intervals are reported; no formal final-test claim.

All selected outcomes were physically executed during the complete matched
collection. This supports exact offline potential-outcome comparison for this
fixed local panel. It does **not** execute a learned policy online, establish
replanning behavior, assess latency under a running robot, or demonstrate memory,
beacon discovery, return, realistic sensing or physical transfer. A policy that
always stops can avoid contact and still fail the navigation task; stop frequency
and regret are mandatory alongside contact counts.
