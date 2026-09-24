# Prospective comparison analysis for the original eight-layout definition

This is a proposed statistical supplement for the existing four-arm, 32-episode
definition. No independent-layout sensor outcomes have been consumed, no final
population review has been written, and no study has been launched by this work.
The eight-diagnostic final review must decide whether to retain that definition
and this analysis plan before outcome access. A changed arm/model/layout roster
requires a corresponding checked analysis successor; this module accepts only
the original fixed roster.

The existing completed-population readout is descriptive. The supplement in
`lewm/independent_round_trip_exact_comparison_analysis_development.py` keeps all
32 cases, including scientific failures, and uses **eight layout units**, not
32 independent trials. It checks the ordered outcome projection and reconstructs
arm and paired success totals. Its caller must authenticate the complete original
readout and its upstream evidence; these structural checks do not replace raw
audits or artifact authentication.

The three hypotheses retain their existing scope:

| Contrast | Permitted interpretation |
| --- | --- |
| Persistent full-RGB JEPA vs supervised rollout | Training objective for the fixed assigned models and seed. |
| Persistent JEPA vs reactive | Whole predictive method vs observed-route method; not isolated prediction ranking. |
| Persistent JEPA vs current-pair JEPA | Accumulated planning cells; contact, tracking and other histories remain. |

For each contrast, use the exact two-sided binomial test on discordant layout
pairs, equivalent to exact McNemar testing. Concordant successes and failures
remain in the eight-layout success-rate denominator but do not supply a
discordance direction. With no discordant pairs, return p=1. The use of a
binomial distribution for the exact paired test is documented by
[statsmodels](https://www.statsmodels.org/v0.14.0/generated/statsmodels.stats.contingency_tables.mcnemar.html).

Apply Holm adjustment to all three predeclared p-values as one family, at
alpha=0.05. Use exact rational arithmetic for probabilities and adjustment;
do not select a smaller family or a one-sided direction after seeing outcomes.
Holm controls family-wise error for valid component p-values without requiring
independence between the three tests. See the
[R documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).

For each arm, report its rate and two-sided 95% Clopper-Pearson interval. Also
report intervals using alpha/4 per arm, providing conservative simultaneous
95% coverage across the four arms under the binomial sampling assumptions.
The implementation inverts finite binomial tails numerically and is checked
against [SciPy's exact binomial intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

The calculations assume independent representative layout draws for binomial
coverage, and independent discordance signs with equal direction probabilities
under each paired null. Disjoint layout identities alone do not establish these
sampling assumptions. The fixed procedural layout design therefore needs an
explicit interpretation in the final review; conditional p-values are not a
randomized-treatment proof or a deployment certificate. The module records
that assumptions are unverified and grants no navigation, timing or hardware
qualification. If any startup is unmatched, withhold all paired inference while
retaining every case and all arm descriptives. Do not drop the problematic pair.

Even hypothetical 8/8 success gives a marginal two-sided 95% exact lower bound
of only 0.6305833524 under those assumptions. With all three contrasts having
8 wins and 0 losses, each raw p-value is 1/128 and each Holm-adjusted p-value
is 3/128. With 7 wins and 1 loss, the raw p-value is 9/128. These are design
calculations, not observed robot results, and emphasize how limited eight
layouts are for broad reliability claims.

Validation: 95 tests passed in 2.45 s, session 96457. All 45 possible discordant
count pairs were compared against SciPy 1.17.1; all nine success counts were
checked for both interval levels. Tests cover Holm ordering, all failures,
all successes, maximal differences, unmatched startups, wrong denominators,
changed rosters and inconsistent summaries. Existing source-bound population
definitions, launch gates, controllers and readouts remain unchanged.
