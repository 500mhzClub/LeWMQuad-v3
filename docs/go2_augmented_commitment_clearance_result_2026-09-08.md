# Executed augmented commitment clearance result

The diagnostic covers all six completed commitments per augmented model and
all 45 completed original full-direct commitments. The augmented no-action
terminal at tick 33 and original interrupted commitment at tick 228 remain
explicitly excluded from complete-commitment labels.

At the failing augmented tick-28 right arc, actual body XY displacement was
[0.084422262, -0.004921482] m. Full direct predicted
[0.064489052, -0.006937444] m (20.035-mm XY error); full JEPA predicted
[0.031672206, 0.020984109] m (58.768-mm error). Their nominal predicted
clearances were 462.129 and 460.918 mm against the same 450-mm requirement.

The cell [11, -2] was already a witness in both models' tick-28 nominal checks.
No future geometry was added. The actual sampled path first crossed the
450-mm nominal radius at within-commitment sample 243 of 250, and eight samples
violated it. Its minimum distance to the recorded witnesses was 448.887 mm;
the observed endpoint's distance was 448.524 mm. Thus the failure is not
explained by a newly seen obstacle. It is a nominal-circle conflict, not a
physical collision; the native physical-stop and contact results are unchanged.

Mean/max XY errors over six commitments were 18.993/22.398 mm for augmented
direct and 51.812/58.968 mm for augmented JEPA. The original direct trajectory's
45 complete commitments had 16.930/37.866 mm error and no sampled conflict
against these limited witnesses. It executed a different action at tick 28;
its result is not a counterfactual right-arc label. No absence of conflict
certifies all occupied cells, continuous articulated motion or unknown space.

Artifact root: `go2_augmented_commitment_clearance_v1_attempt_001`, under the
established navigation development artifact root. Launch SHA-256:
`379627dd395b74c434f225362d4b40acf693ee61e5d36f4901f3fad713fcf91d`.
Result SHA-256:
`991feea7a987e9c75027453f5f2d9a8dcd7034e2d7a14bb722841a53eb8c292c`.
The run binds 1,087 source paths and took 11.587 seconds after launch. Two
focused path-accounting tests passed. There was no execution, fitting,
threshold change, checkpoint choice or revision of any earlier outcome.
