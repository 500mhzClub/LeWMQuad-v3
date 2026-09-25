"""Report saved V4.2 outputs only. No physics, model calls, fitting or reranking."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

from lewm import decision_headroom_json_v42_development as output_json
from scripts.read_go2_headroom_v42_development import layout_interval

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_decision_headroom_phase2_v4_attempt_001')
DOCS = Path('docs').absolute()
SCOPES = ('exposed', 'runtime_unexamined_at_freeze')
SOURCES = ('command history', 'reactive feedback', 'learned / old head')


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)


def estimate(value, scale=1):
    mean, interval = value.get('mean'), value.get('interval')
    if mean is None:
        return 'unavailable (n=0)'
    text = f'{scale*mean:.4f}'
    return text + (f' [{scale*interval[0]:.4f}, {scale*interval[1]:.4f}]' if interval else ' [no interval]') + f"; n={value['n']}"


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows]) + '\n'


def main():
    output_json.install(DOCS)
    analysis, resources = read(ROOT/'analysis_v42.json'), read(ROOT/'resource_result.json')
    config = read(DOCS/'go2_decision_headroom_protocol_v42_2026-09-23.json')
    collection = read(ROOT/'collection_result.json')
    assert resources['error'] is None and not resources['resource_stop_latched']
    assert len(collection['cells']) == 24 and analysis['primary_family_size'] == 13
    records, cell_rows, cases = [], [], {}
    validity = collections.Counter()
    modes, safety, pose = collections.Counter(), collections.Counter(), collections.Counter()
    rgb_checks = rgb_matches = physical_replays = repeat_checks = 0
    for case in range(24):
        source = ROOT/f'source_{case:02d}'
        states = read(source/'snapshots.json')
        rows = []
        for state in states:
            path = source/f"state_{state['frame']:04d}"/'audit_v4.json'
            if not path.exists():
                continue
            row = read(path)
            rows.append(row)
            records.append((case, row))
            for key in ('source_input_valid', 'physics_valid', 'rgb_valid'):
                validity[key] += int(row[key])
            modes[row['objective']['mode']] += 1
            pose['position/'+row['localisation'].get('position_stratum', 'unresolved')] += 1
            pose['yaw/'+row['localisation'].get('yaw_stratum', 'unresolved')] += 1
            for candidate in row['safety']:
                for criterion in ('hard', 'operating'):
                    safety[criterion+'/'+candidate[criterion]] += 1
                safety['contact'] += int(candidate['contact'])
            restoration = read(path.parent/'restoration.json')
            for comparison in restoration['comparisons']:
                physical_replays += 1
                if comparison['kind'] == 'source':
                    rgb_checks += len(comparison['rgb'])
                    rgb_matches += sum(p['bitwise_equal'] for p in comparison['rgb'])
                else:
                    repeat_checks += 1
        positional = [r for r in rows if r['objective']['reference_regret_applicable']]
        weight = sum(r['sampling']['weight'] for r in positional)
        available = {k: sum(r[k]['status'] == 'available' for r in positional) for k in ('reference', 'secondary_reference')}
        coverage = {k: sum(r['sampling']['weight'] for r in positional if r[k]['status'] == 'available')/weight if weight else None for k in available}
        source_result = read(source/'result.json') if (source/'result.json').exists() else {}
        mission = read(source/'mission.json')
        cases[case] = dict(audited=len(rows), positional=len(positional), view=len(rows)-len(positional),
                           available=available, weighted_reference_coverage=coverage,
                           controller_terminal=mission[-1].get('terminal'),
                           source_wall_s=source_result.get('wall_s'), source_policy_steps=source_result.get('policy_steps'))
        cell_rows.append([f'{case//3:02d}', SOURCES[case%3], len(rows), len(positional),
                          available['reference'], available['secondary_reference'],
                          ' / '.join('N/A' if coverage[k] is None else f'{coverage[k]:.1%}' for k in available)])
    assert len(records) == sum(c['retained_audit_records'] for c in analysis['cell_coverage'])
    assert rgb_checks == rgb_matches
    primary = {scope: {q: analysis['layout_clustered'][f'{scope}/all/all/{q}'] for q in analysis['primary_quantities']} for scope in SCOPES}
    memo_tests = {}
    for scope in SCOPES:
        value = lambda q: analysis['layout_clustered'][f'{scope}/all/all/{q}']
        lower = lambda q: value(q)['interval'][0] if value(q).get('interval') else None
        r5 = lower('paired_filter/R5c-R2/all_excluded_despite_safe')
        tests = {head: None if r5 is None else lower(f'paired_filter/R4/{head}-R2/all_excluded_despite_safe') > .10 and r5 <= .10 for head in ('old_data','maze_data')}
        tests['B'] = lower('memo/R2/motion_binding_complete_exclusion') > .10 or r5 > .10
        tests['restricted_R2'] = value('memo/R2/motion_binding_complete_exclusion')
        memo_tests[scope] = tests
    # Prespecified secondary quantities already exist per cell. Apply the same
    # equal-source/layout estimator at 95%; no new selections or reference costs.
    secondary = {}
    for scope, layouts in zip(SCOPES, (range(4), range(4,8))):
        values = {}
        for head in ('old_data', 'maze_data'):
            for component in ('G','H_scorer','H_motion','D','L_readout','L_forecast'):
                key = f'chain/{head}/{component}'
                layout_values = []
                for layout in layouts:
                    cell_values = [analysis['cells'][f'{3*layout+j}/all/all']['chain/'+head]['components'][component]['value'] for j in range(3)]
                    layout_values.append(statistics.mean(cell_values) if all(v is not None for v in cell_values) else None)
                values[key] = layout_interval(layout_values, .95)
            suffix = 'old' if head == 'old_data' else 'maze'
            for q in ('A_action_'+suffix, 'shrinkage_'+suffix):
                layout_values = []
                for layout in layouts:
                    cell_values = [analysis['cells'][f'{3*layout+j}/all/all'][q]['value'] for j in range(3)]
                    layout_values.append(statistics.mean(cell_values) if all(v is not None for v in cell_values) else None)
                values[q] = layout_interval(layout_values,.95)
        secondary[scope] = values
    positional = [r for _,r in records if r['objective']['reference_regret_applicable']]
    reference_counts = {key: sum(r[key]['status']=='available' for r in positional) for key in ('reference','secondary_reference')}
    clearance_coverage = collections.defaultdict(collections.Counter)
    for row in positional:
        for i, stratum in enumerate(row['endpoint_clearance_strata']):
            counts = clearance_coverage[stratum]
            counts['candidates'] += 1
            for label, field, reference in (('primary','costs','reference'),('secondary','secondary_costs','secondary_reference')):
                counts[label+'_cost'] += len(row[field]) > i and row[field][i].get('cost_s') is not None
                counts[label+'_reference'] += row[reference]['status'] == 'available'
    historical_path = ROOT.parent/'go2_headroom_historical_rerender_v42_attempt_001'/'result.json'
    historical = read(historical_path)
    identities = {name: binding(ROOT/name) for name in ('collection_result.json','analysis_v42.json','branch_panel_v42.json','memo_inputs_v42.json','resource_result.json','handover_boundary.json','resume_admission.json')}
    identities['historical_rerender'] = binding(historical_path)
    resources_short = {k:v for k,v in resources.items() if k not in ('measurements','cache_baseline_bytes')}
    summary = dict(schema='headroom_v42_scientific_closeout.v1', status='CHECKPOINT_B_STOP',
        protocol_sha256=binding(DOCS/'go2_decision_headroom_protocol_v42_2026-09-23.json')['sha256'],
        audit_artifacts=identities, source_assignments=collection['cells'], cases=cases,
        audited_states=len(records), planned_states=576, validity=dict(validity), objective_modes=dict(modes),
        safety=dict(safety), localisation_counts=dict(pose), source_rgb_matches=rgb_matches,
        source_rgb_comparisons=rgb_checks, restoration_comparisons=physical_replays,
        candidate_repeat_comparisons=repeat_checks, reference_available_counts=reference_counts,
        primary=primary, secondary_common_population=secondary, memo_tests=memo_tests,
        resources=resources_short, historical_rerender_status=historical['status'],
        conclusion='Primary regret and attribution rules inconclusive; coverage and layout precision limit interpretation.',
        recommendation='Propose one bounded, preregistered follow-up audit design addressing positional-reference coverage and independent-layout precision; do not execute or fit.',
        recommendation_executed=False, further_physics_authorized=False)
    summary_path = DOCS/'go2_decision_headroom_audit_result_2026-09-25.json'
    with summary_path.open('x') as stream:
        json.dump(summary,stream,indent=2)
        stream.write('\n')
    text = ['# Decision-headroom audit V4.2 — final result, 2026-09-25',
        '**Decision-table outcome: measurement/coverage/precision limitation; scientific verdict inconclusive.** Neither predeclared filter-attribution condition A nor B is supported in either exposure stratum. No primary regret quantity establishes command-history sufficiency, a learned benefit, or a representation/predictor defect. Inconclusive regret supports neither stopping nor continuing visual ego-motion research. The authorized audit is complete and stops at checkpoint (b).',
        'The point estimates favour command history over both learned heads, but all primary regret intervals cross zero and the ±0.02-s practical-effect region. These are local 800-ms cost differences, not measured mission-time savings. The secondary-reference findings below are descriptive and do not override the primary result.',
        '## Primary results',
        'Means and frozen 99.6154% intervals (1 − 0.05/13), with the number of contributing layout clusters. Rates are percentage points; regret quantities are seconds. Intervals are printed untruncated, including negative bounds for nonnegative quantities and bounds outside rate support. Such wide t intervals diagnose poor precision. `D_old` and `D_maze` are learned minus command-history regret: negative would favour learned prediction.',
        table(['Quantity','Exposed layouts 00–03','Runtime-unexamined-at-freeze layouts 04–07'],
              [[q,estimate(primary[SCOPES[0]][q],1 if q in ('G','H_scorer','H_motion','D_old','D_maze') else 100),estimate(primary[SCOPES[1]][q],1 if q in ('G','H_scorer','H_motion','D_old','D_maze') else 100)] for q in analysis['primary_quantities']]),
        '`G` is command-history regret against the reference; `H_scorer` is regret with true motion; `H_motion` is command history minus true-motion regret. They use their frozen applicable populations; the separate common-mask decomposition below is used for telescoping, not subtraction of differently masked primary means.',
        '## Completed scope and validity',
        f"24 fixed source assignments closed: 23 completed and case 13 (layout 04/reactive) unresolved after `measured visual pose unavailable`. Its 24 sampled identities and failure remain in the panel with missing audit outputs. There were no retries or substitutions. {len(records)}/576 sampled states have audit records ({len(records)/576:.1%}). All {len(records)} passed source decision-input reproduction, physical restoration and RGB qualification. Source-replay RGB matches: {rgb_matches:,}/{rgb_checks:,}. Candidate-repeat comparisons: {repeat_checks}. The 5,244 branches include restoration checks and predetermined repeats; they are not 5,244 independent decision states.",
        f"The sampled modes are {modes['positional_route']} route-target, {modes['positional_terminal']} terminal-target and {modes['view_seeking']} target-free view-seeking states. The latter remain in the filter audit and have no invented positional regret.",
        f"All {safety['operating/safe']:,} first-repeat bank candidate outcomes (six per audited state; 2,760 movement candidates) were classified safe under both the 5-mm/contact hard criterion and the 20-mm operating margin. No disallowed contact was recorded in those outcomes. Therefore unsafe-candidate discrimination is untested: zero admitted-unsafe observations do not establish a gate's sensitivity to danger.",
        'Clearance uses all 27 collision primitives at native 2-ms steps, with feet/legs/body included and ordinary support contact excluded. The FK endpoint-displacement robustness calculation also passed for these rows. As frozen in V4, this is native discrete ground truth plus a reported interval robustness convention; it is not certification of arbitrary unobserved continuous articulated motion. Original analytical fixture failures, erratum and 13 passing boundary cases remain in the earlier Phase 1 records; analytical tests are not substituted for these trajectories.',
        'The frozen harm panel reports zero means and degenerate [0,0] layout-t intervals for the tested selected actions. Zero observed harm and zero between-layout variation do not prove zero risk or certify the 5% absolute / 1% excess-harm limits with adequate rare-event uncertainty. No safety or learned-benefit claim is made from those intervals.',
        '## Reference coverage and denominators',
        f"There are {len(positional)} positional states. Primary reference availability is {reference_counts['reference']}/{len(positional)} ({reference_counts['reference']/len(positional):.2%}); secondary availability is {reference_counts['secondary_reference']}/{len(positional)} ({reference_counts['secondary_reference']/len(positional):.2%}). These are raw counts, not pooled substitutes for the declared weighted estimator. The frozen quantity-coverage threshold is 90%; multiple source cells fall below it. Primary regret has complete three-source support in only two exposed and three initially unexamined layout clusters. Layouts 02 and 03 have reactive samples entirely in view-seeking mode; layout 04 lacks its reactive audit. No missing source is reweighted away.",
        'The secondary reference restores some physically safe endpoints inside the 0.46-m inflation region by the approved nearest-free-cell path convention. It does not change actual candidate safety, controller eligibility, or the primary reference. Remaining bank-cost unresolved states are retained. Operating-margin cost is zero on every available reference here because every bank candidate satisfied the margin; rejecting margin-satisfying candidates is a different cost and remains in per-row `excess_rejection_cost_s`.',
        'Endpoint-clearance coverage below is a raw candidate count over positional states, not a weighted performance estimator. All these candidate trajectories passed the articulated operating margin; endpoint strata refer to the separate conservative 0.46-m reference inflation. Reference-available counts mean that the candidate belongs to a state with an available bank optimum.',
        table(['Reference endpoint stratum','Candidates','Primary finite endpoint cost','Secondary finite endpoint cost','Primary reference available','Secondary reference available'],[[stratum]+[clearance_coverage[stratum][k] for k in ('candidates','primary_cost','secondary_cost','primary_reference','secondary_reference')] for stratum in ('reference_free','inside_046m_inflation','outside_inflation_below_5mm')]),
        table(['Layout','Source','Audited','Positional','Primary available','Secondary available','Weighted coverage primary / secondary'],cell_rows),
        'The full endpoint-cost and reference availability breakdown by clearance stratum, hard/operating status and case is in `analysis_v42.json:reference_coverage_by_clearance`. The full state/source/phase/localisation weighted denominators and exclusion counts are in `analysis_v42.json:cells`. No imputed rows or modified common masks are used.',
        '## Filter attribution and binding rules',
        'True-motion R2 still excludes 56.08% of safe movement candidates on exposed layouts and 40.35% on initially unexamined layouts (layout means). It excludes every movement despite an available safe movement on 40.82% and 22.22% of states respectively. These totals include observation-only view restrictions as well as motion-dependent rules; the unrestricted totals cannot establish condition B.',
        table(['Scope','A: old head','A: maze head','B','Restricted R2 binding level, % [99.6154% interval]'],
              [[scope,str(memo_tests[scope]['old_data']),str(memo_tests[scope]['maze_data']),str(memo_tests[scope]['B']),estimate(memo_tests[scope]['restricted_R2'],100)] for scope in SCOPES]),
        'Neither learned-head complete-exclusion contrast has a lower bound above τ=10 percentage points. Neither the restricted R2 motion-binding level nor the R5c contrast clears B. This is failure to establish the predeclared attribution, not evidence that either mechanism is absent. The exploratory Stage A hold analysis motivated this component; it did not determine cost weights or select a favourable comparison scope.',
        '## Descriptive diagnostics',
        'The following 95% intervals use the same equal-source/layout estimator on the already-written per-cell quantities. The chain uses the identical R2/R3/R4/R5c state mask and weights within each head. `L_readout` and `L_forecast` are arithmetic pathway differences, not causal attributions. Action-derangement and shrinkage comparisons retain their own masks.',
        table(['Secondary quantity, seconds','Exposed','Initially unexamined'],[[q,estimate(secondary[SCOPES[0]][q]),estimate(secondary[SCOPES[1]][q])] for q in secondary[SCOPES[0]]]),
        table(['Secondary-reference regret, seconds (95%)','Exposed','Initially unexamined'],[[q,estimate(analysis['layout_clustered'][f'exposed/all/all/secondary_reference/{q}']),estimate(analysis['layout_clustered'][f'runtime_unexamined_at_freeze/all/all/secondary_reference/{q}'])] for q in ('G','H_scorer','H_motion','D_old','D_maze')]),
        'In the secondary reference the initially unexamined learned-minus-command-history intervals are positive for both heads. This is a descriptive conditional result, not a replacement for the inconclusive primary family or a representation-defect claim. Common-mask chain details, reactive-row bank membership, optimal-set membership, unnecessary holds and costs of eligibility rejection remain in the versioned per-state panel. No off-bank branch was required in this run.',
        '### Per-motion-source filter levels (descriptive 95%)',
        table(['Motion source / quantity, %','Exposed','Initially unexamined'],[[motion+'/'+q,estimate(analysis['layout_clustered'][f'descriptive/exposed/all/all/filter/{motion}/operating/{q}'],100),estimate(analysis['layout_clustered'][f'descriptive/runtime_unexamined_at_freeze/all/all/filter/{motion}/operating/{q}'],100)] for motion in ('R2','R5c','R4/old_data','R4/maze_data') for q in ('excluded_safe','all_excluded_despite_safe','admitted_unsafe')]),
        '### Localisation covariate',
        table(['Raw audited-state stratum','Count'],sorted(pose.items())),
        'The following saved descriptive panels retain their own source/layout support. Covariate stratification is observational and does not identify a localisation effect.',
        table(['Scope / position stratum','G, seconds (95%)','Old-head complete-exclusion difference, pp (95%)','Maze-head complete-exclusion difference, pp (95%)'],
              [[scope+'/'+mode,estimate(analysis['layout_clustered'][f'{scope}/pose_position/{mode}/all/G']),estimate(analysis['layout_clustered'][f'{scope}/pose_position/{mode}/all/paired_filter/R4/old_data-R2/all_excluded_despite_safe'],100),estimate(analysis['layout_clustered'][f'{scope}/pose_position/{mode}/all/paired_filter/R4/maze_data-R2/all_excluded_despite_safe'],100)] for scope in SCOPES for mode in ('le_20mm','20_to_100mm','gt_100mm','unresolved')]),
        'Yaw strata and route/terminal/view and mission-phase panels are retained in the saved analysis. Absence of sufficient layouts in a stratum remains unavailable; no strata are pooled after seeing results.',
        '## Stage A holds and historical training provenance',
        'Stage A completed unchanged before this audit. These are exploratory counts from different trajectories, not isolated readout effects:',
        table(['Run','Hold plans','No eligible movement','Eligible but outscored/tied','Override','Insufficient'],[['00/old',987,898,88,1,0],['00/maze',979,875,19,84,1],['02/maze',1140,1138,1,1,0],['02/old',957,614,97,244,2]]),
        'Observation-only / motion-clearance / stopping-projection flags were respectively 1/899/2, 837/959/0, 1131/1139/1, and 293/859/12 in the same run order. These overlapping flags do not sum to holds. See [the unchanged hold analysis](go2_stage_a_holds_exploratory_2026-09-23.md) for dispatch categories and retained-evidence limitations.',
        'The separately approved historical check selected 16 training examples (four per population), inspected 32 source frames and rendered zero frames: all examples lacked a retained bound restore packet for the qualified path. Its status is `COMPLETE_WITH_UNRESOLVED_INPUTS`, not a bitwise pass or demonstrated corruption. It did not gate this audit and no historical physics or regeneration was performed.',
        table(['Historical population/result with residual uncertainty','Current predictor','Old-data readout','Maze-data readout'],[
            ['Factorial predictor / V1.2 ancestry','Inherited checkpoint/training ancestry','No direct image/weight edge identified; deployment receives predictor forecasts','Same deployment dependence'],
            ['Older native geometry-progress and moving-action-switch training','Direct training inputs','Actual RGB pairs and inherited motion-head weights','Same old-data ancestry'],
            ['Full-heading continuation','No new predictor fitting','Inherited mixed-data head and heading pairs','Same initialization and old pairs'],
            ['Maze-view training/recovery','Frozen; no updates','New maze images not optimizer examples','Direct maze-view training pairs'],
            ['Near-goal / stalled-turn / frozen-native comparisons','Evaluation, not fitting from these assays','Evaluation, not training populations from those diagnoses','Same distinction'],
            ['Original failed pilot branch futures','No training or comparative audit dependency','None','None']]),
        'The scoped source-path conclusions and exact historical result identities remain in [the provenance report](go2_renderer_provenance_readonly_2026-09-23.md) and [training-dependency addendum](go2_remaining_phase1_provenance_addendum_2026-09-23.md). Training render validity for the current predictor and both readouts remains unverified. Consequently pathway loss cannot be specifically attributed to representation defects. That caveat does not invalidate this frozen-model audit with newly qualified inputs/outcomes. It establishes neither historical corruption nor proven absence of exposure.',
        '## Resource use and retained assets',
        table(['Quantity','Measured','Cap'],[
            ['Source attempts',24,24],['Sampled / audited states','576 / 552',576],['Attempted branches',resources['branch_attempts'],6096],
            ['Source simulated seconds',resources['reserved_source_physics_s'],11556],['Branch simulated seconds',resources['reserved_branch_physics_s'],4876.8],
            ['Wall hours',f"{resources['wall_s']/3600:.3f}",72],['CPU core-hours',f"{resources['cpu_s']/3600:.3f}",1152],
            ['Peak sampled RAM GiB',f"{resources['peak_sampled_aggregate_rss_bytes']/1024**3:.3f}",32],['Peak sampled VRAM GiB',f"{resources['peak_sampled_total_gpu_used_bytes']/1024**3:.3f}",8],
            ['Retained GiB at resource closeout',f"{resources['retained_bytes']/1024**3:.3f}",20]]),
        'No resource stop or JSON writer failure occurred. Resource peaks are sampled observations, not OS-enforced hard limits. The handover preserved the original assignment prefix and cumulative budgets. The final converter receipt covers successor writes; predecessor immediate checks were unchanged, but its in-memory receipt was not transferred. Source navigation outcomes are descriptive, not a new navigation cohort result.',
        f"The versioned branch panel is `{ROOT/'branch_panel_v42.json'}` (SHA-256 `{identities['branch_panel_v42.json']['sha256']}`). It indexes all 576 sampled identities, including 24 missing-output rows, and links the retained source packets, physical trajectories, render checks, row masks and costs. Large runtime artifacts remain outside Git. [The compact closeout JSON](go2_decision_headroom_audit_result_2026-09-25.json) binds the analysis, panel, resources and historical check by SHA-256.",
        'Authority: [V4.2 protocol](go2_decision_headroom_protocol_v42_2026-09-23.json), [explicit approval](go2_decision_headroom_v42_approval_2026-09-23.json), and [monitoring-only handover amendment](go2_decision_headroom_v42_monitor_handover_2026-09-24.json). [Decision memo](go2_decision_headroom_decision_memo_2026-09-25.md). No follow-up experiment, fitting, gate repair, sample expansion or artifact retirement is executed.']
    report = DOCS/'go2_decision_headroom_audit_result_2026-09-25.md'
    with report.open('x') as stream:
        stream.write('\n\n'.join(text)+'\n')
    memo = '''# Decision memo — V4.2, 2026-09-25

**Decision: inconclusive; apply the handoff's measurement/coverage/precision-limitation row. Stop the authorized audit at checkpoint (b).**

All 24 fixed source assignments are closed, with one preserved tracking failure. The audit produced 552 qualified states and 5,244 branch attempts. Source inputs, physical restoration and RGB comparisons passed for all audited states. This establishes a usable frozen-model diagnostic panel, not a JEPA advantage.

Primary 99.6154% regret intervals, in seconds:

'''+table(['Quantity','Exposed (2 contributing layouts)','Initially unexamined (3 contributing layouts)'],[[q,estimate(primary[SCOPES[0]][q]),estimate(primary[SCOPES[1]][q])] for q in ('G','H_scorer','H_motion','D_old','D_maze')])+'''
The practical-effect threshold is δ=0.02 s. None of these intervals supports material superiority, equivalence, or command-history near-optimality. Positive learned-minus-command-history point estimates are not a statistically established JEPA loss. Inconclusive regret supports neither stopping nor continuing visual ego-motion optimisation. Local costs cannot be summed into predicted mission-time savings without additional assumptions.

The filter audit shows substantial exclusion of safe movement even with true motion: R2 excluded-safe levels are 56.08% (exposed) and 40.35% (initially unexamined). Complete-exclusion levels are 40.82% and 22.22%. However, these include observation-only rules. Neither condition A (learned-motion-specific false exclusions) nor condition B (motion-gate conservatism independent of prediction quality) meets its frozen lower-bound test at τ=0.10 in either stratum. The restricted R2 motion-binding intervals are −38.42 to 69.67 percentage points and −125.33 to 147.55 percentage points. Do not choose a gate repair from the unrestricted totals.

The primary reference covers 337/417 positional states (80.82% unweighted); the secondary reference covers 375/417 (89.93%). Several source cells fail the frozen 90% quantity-coverage criterion. Two reactive source cells contain only target-free view-seeking states, and the failed reactive cell removes another complete layout cluster. The secondary reference recovers some coverage but does not rescue the primary inference. All tested bank candidates were safe over the short branch horizon, leaving unsafe-candidate discrimination untested. Degenerate zero-harm t intervals are not proof of zero risk.

**Single recommended next step, requiring separate approval:** prepare one bounded, preregistered revision of the decision-headroom study design that can meet positional-reference coverage and independent-layout precision requirements. Keep the current controller/model comparison fixed while specifying the measurement design; do not start another fitting or gate-tuning cycle.

The proposed preregistered evaluation should fix positional-objective/source quotas, a physically interpretable reference with a declared coverage gate, independent-layout sample size justified against δ=0.02 s, and a valid rare-harm uncertainty rule before new comparative data. Preserve the separation of physical safety, operating margin and controller eligibility. Fix maximum collection/compute/storage budgets and retain an inconclusive outcome when those budgets cannot support the precision target. Any new sample or metric constitutes a separately approved study, not an extension or replacement of this audit. This memo proposes that design work; it does not perform it or choose new quotas, thresholds, layouts or costs.

Historical training rendering remains unverified: the separate 16-example check could not rerender any sample through the qualified path because bound restoration packets were unavailable. L_readout/L_forecast localise arithmetic pathway differences only; they cannot isolate a representation defect from training-provenance or scorer interactions.

The full primary/secondary tables, source-cell denominators, paired masks, descriptive localisation and filter panels, Stage A holds, training-dependency qualifications, asset hashes and resource use are in [the result report](go2_decision_headroom_audit_result_2026-09-25.md). The versioned V4.2 branch panel is retained and bound by [the closeout JSON](go2_decision_headroom_audit_result_2026-09-25.json).

**No recommendation has been implemented. No further experiment is running or authorized by this closure.**
'''
    with (DOCS/'go2_decision_headroom_decision_memo_2026-09-25.md').open('x') as stream:
        stream.write(memo)
    print(json.dumps(dict(report=str(report),audited_states=len(records),rgb_matches=rgb_matches,
                          memo_tests=memo_tests,checkpoint='b',stopped=True),indent=2))


if __name__ == '__main__':
    main()
