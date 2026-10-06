"""Static dependency closure of the navigation-capability programme's entry points (consolidation, 6 October 2026).

For the research-artefact migration plan (docs/go2_research_artifact_migration_plan_2026-10-06.md). Starting from the
entry points listed below, it follows, in tracked repository Python files only:
- every `import` / `from ... import` statement anywhere in a file (including inside functions);
- string constants naming a repository module (`lewm.x`, `scripts.y`) or a repository file path (`scripts/z.py`,
  `config/...`, `docs/...`), which covers importlib, runpy and file-path loads.
Shell scripts (*.sh) are followed too: their `$SCRIPT_DIR/x`, `$ROOT/scripts/x` and `scripts/x` references, and
`python[3] -m module` invocations. Non-Python files reached through string constants (configs, docs, JSON bindings) are listed but not parsed.
String constants that look like paths outside the repository (absolute paths, or paths under the artefact root) are
collected as artefact references.

**Custody.** Candidate files come from `git ls-files` (names only). Any path matching the sealed patterns
(`sealed_test.json`, a `sealed/` directory, a `sealed_*` directory) is dropped by name before anything is opened, and is
reported only as an excluded reference. No file under such a path is ever read.

Output: JSON to stdout or --out (closure files, per-entry counts, unresolved module names, artefact references).
"""
import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = {'lewm': 'lewm', 'scripts': 'scripts', 'lewm_genesis': 'lewm_genesis/lewm_genesis',
                 'lewm_worlds': 'lewm_worlds/lewm_worlds', 'lewm_go2_control': 'lewm_go2_control/lewm_go2_control'}
SEALED = re.compile(r'(^|/)(sealed_test\.json$|sealed/|sealed_[^/]*/)')
MODULE_STRING = re.compile(r'^(lewm|scripts|lewm_genesis|lewm_worlds|lewm_go2_control)(\.[A-Za-z_][A-Za-z0-9_]*)+$')
PATH_STRING = re.compile(r'^[A-Za-z0-9_./-]+\.(py|sh|json|yaml|yml|md|txt|npz|pt|csv|sha256|png|jpg|obj|urdf|xml)$')

LAUNCHERS = sorted(str(p.relative_to(REPO)) for p in (REPO/'scripts').glob('launch_go2_dev_cohort_pinned*_development.py'))
ENTRIES = sorted(str(p.relative_to(REPO)) for p in (REPO/'scripts').glob('run_go2_dev_mission_pinned*_development.py'))
ENTRY_POINTS = {
    'maze and scene generation': [
        'scripts/generate_go2_navigation_capability_sets_development.py', 'scripts/register_go2_c3v2_sets_development.py',
        'scripts/register_go2_c3v3_sets_development.py', 'lewm_worlds/lewm_worlds/corpus.py'],
    'locomotion RL training': [
        'scripts/train_genesis_go2_locomotion_contract.py', 'scripts/train_genesis_go2_locomotion_contract.sh',
        'scripts/fetch_genesis_go2_locomotion_examples.sh', 'scripts/check_genesis_go2_policy_contract.py',
        'scripts/check_genesis_go2_policy_contract.sh', 'scripts/setup_genesis_rocm_training.sh'],
    'May-corpus data generation': [
        'scripts/datagen_all_resumable.sh', 'scripts/datagen_render_resumable_v03.sh', 'scripts/render_replay_v03.py'],
    'T1 temporal predictor': [
        'scripts/run_dev_proprio_factorial_driver_v1.py', 'scripts/run_dev_v03_temporal_action_jepa_v1.py',
        'scripts/eval_dev_v03_temporal_action_jepa_v1.py', 'scripts/complete_dev_v03_temporal_action_jepa_evaluation_v1.py',
        'scripts/run_dev_v03_two_step_rollout_v1.py', 'scripts/build_dev_v03_proprio_action_manifest_v1.py',
        'scripts/build_dev_canonical_cache_map_v1.py', 'scripts/build_dev_factorial_manifest_v1.py',
        'scripts/freeze_dev_proprio_run_package_v1.py', 'scripts/eval_dev_proprio_factorial_v1.py'],
    'T2-T5 predictor and readout training': [
        'scripts/train_go2_frozen_vjepa_native_adaptation_development.py', 'scripts/evaluate_go2_frozen_vjepa_native_adaptation_development.py',
        'scripts/train_go2_balanced_start_predictor_development.py', 'scripts/train_go2_dense_task_predictor_development.py',
        'scripts/train_go2_horizon_dense_predictor_development.py', 'scripts/train_go2_full_heading_readout_development.py',
        'scripts/train_go2_maze_view_readout_development.py', 'scripts/run_go2_maze_view_readout_recovery_development.py',
        'scripts/train_go2_dense_visual_motion_readout_development.py'],
    'training data collection': [
        'scripts/collect_go2_balanced_start_actions_development.py', 'scripts/collect_go2_balanced_start_horizon_actions_development.py',
        'scripts/run_go2_moving_action_switch_family_v1.py', 'scripts/run_go2_geometry_progress_family_v1.py',
        'scripts/collect_go2_short_pulse_learning_development.py', 'scripts/collect_go2_full_heading_training_development.py',
        'scripts/collect_go2_maze_view_training_development.py', 'scripts/collect_go2_maze_view_transfer_development.py',
        'scripts/collect_go2_c3v2_rest_turn_recordings_development.py', 'scripts/run_go2_c3v3_round_cohort_development.py',
        'scripts/run_go2_c3v3_round_development.py', 'scripts/replay_go2_c3v3_onpolicy_frames_development.py',
        'scripts/prepare_go2_short_pulse_training_development.py', 'scripts/derive_go2_all_phase_training_targets_v1.py',
        'scripts/derive_go2_pre_switch_training_targets_development.py', 'scripts/pre_switch_training_data_development.py'],
    'C1 lineage': ['scripts/fit_go2_local_motion_controls_development.py'],
    'historical C3/C4 stages': ['scripts/train_go2_c3v3_readout_development.py', 'scripts/train_go2_c4v3_development.py',
                                'scripts/train_go2_c3v2_readout_development.py', 'scripts/train_go2_c4v2_development.py'],
    'all pinned launchers and entries': LAUNCHERS+ENTRIES,
    'experiment reports': [
        'scripts/calibrate_go2_forecast_margins_development.py', 'scripts/analyse_go2_forecast_sensitivity_close_approaches_development.py',
        'scripts/analyse_go2_forecast_sensitivity_error_budget_development.py', 'scripts/diagnose_go2_forecast_sensitivity_failures_development.py',
        'scripts/report_go2_reserve_exit_rerun_development.py', 'scripts/render_go2_capability_v4_video_development.py'],
    'missions (final pinned launch)': [
        'scripts/launch_go2_dev_cohort_pinned_v14_development.py', 'scripts/run_go2_dev_mission_pinned_v14_development.py',
        'scripts/run_go2_dev_cohort_development.py', 'scripts/read_go2_dev_mission_development.py'],
    'maze sets': ['scripts/generate_go2_navigation_capability_sets_development.py',
                  'scripts/register_go2_stage2_patch_sets_development.py'],
    'C3/C4 data and decoder': [
        'scripts/prepare_go2_c3v2_c4v2_data_development.py', 'scripts/prepare_go2_c3v3_onpolicy_data_development.py',
        'scripts/build_go2_dev_c3_feature_cache_development.py', 'scripts/fit_go2_dev_decoder_development.py',
        'scripts/select_go2_dev_decoder_development.py'],
    'frozen JEPA model lineage': ['lewm/dense_horizon_navigation_development.py'],
    'C1 command model': ['scripts/fit_go2_short_pulse_command_control_development.py'],
    'results and reports': [
        'scripts/report_go2_prelim_results_development.py', 'scripts/summarise_go2_dev_cohorts_development.py',
        'scripts/report_go2_forecast_sensitivity_development.py', 'scripts/score_go2_dev_closed_loop_prediction_development.py'],
    'videos': ['scripts/render_go2_prelim_video_development.py', 'scripts/render_go2_hero_video_development.py'],
    'dynamics stage 1': ['scripts/characterise_go2_friction_open_loop_development.py'],
    'dynamics stage 2': [
        'scripts/replay_go2_stage2_recording_frames_development.py', 'scripts/count_go2_stage2_recording_contexts_development.py',
        'scripts/score_go2_dev_patch_edge_prediction_development.py', 'scripts/probe_go2_marker_visibility_development.py',
        'scripts/build_go2_stage2_feature_cache_development.py', 'scripts/fit_go2_stage2_decoder_development.py',
        'scripts/fit_go2_stage2_c1_refit_development.py', 'scripts/render_go2_stage2_unmarked_frames_development.py',
        'scripts/build_go2_stage2_unmarked_eval_cache_development.py', 'scripts/evaluate_go2_stage2_marker_control_development.py'],
}


def tracked():
    names = subprocess.run(['git', 'ls-files'], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    return {n for n in names if not SEALED.search(n)}, {n for n in names if SEALED.search(n)}


def module_file(name, files):
    parts = name.split('.')
    root = PACKAGE_ROOTS.get(parts[0])
    if root is None:
        return None
    base = PurePosixPath(root, *parts[1:])
    for candidate in (f'{base}.py', f'{base}/__init__.py'):
        if candidate in files:
            return candidate
    return None


SHELL_REF = re.compile(r'(?:\$SCRIPT_DIR|\$\{SCRIPT_DIR\}|\$ROOT/scripts|\$\{ROOT\}/scripts|(?<![A-Za-z0-9_/])scripts)/([A-Za-z0-9_.-]+\.(?:py|sh))')
SHELL_MODULE = re.compile(r'python[0-9.]*\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)')


def shell_references(path):
    text = (REPO/path).read_text(errors='replace')
    return {f'scripts/{m}' for m in SHELL_REF.findall(text)}, set(SHELL_MODULE.findall(text))


def data_references(path, files):
    """Repository paths named anywhere inside a JSON or YAML data file (bindings, pins, protocols)."""
    import yaml
    full = REPO/path
    if full.stat().st_size > 10_000_000:
        return set()
    try:
        value = json.loads(full.read_text()) if path.endswith('.json') else yaml.safe_load(full.read_text())
    except Exception:
        return set()
    found, stack = set(), [value]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            stack.extend(v.keys())
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
        elif isinstance(v, str) and len(v) < 300:
            candidate = v.split('LeWMQuad-v3/', 1)[-1] if v.startswith('/') else v
            if candidate in files:
                found.add(candidate)
    return found


def references(path, files):
    if path.endswith(('.json', '.yaml', '.yml')):
        return set(), data_references(path, files)
    if not path.endswith(('.py', '.sh')):
        return set(), set()
    if path.endswith('.sh'):
        targets, modules = shell_references(path)
        return modules, targets
    tree = ast.parse((REPO/path).read_text(), filename=path)
    modules, strings = set(), set()
    package = '.'.join(PurePosixPath(path).with_suffix('').parts[:-1])
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                base = '.'.join(package.split('.')[:len(package.split('.'))-node.level+1]+([base] if base else []))
            modules.add(base)
            modules.update(f'{base}.{a.name}' for a in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 300:
            strings.add(node.value)
    return modules, strings


def closure(entries, files, sealed):
    seen, queue, unresolved = set(), list(entries), set()
    data, artefacts, sealed_refs = set(), set(), set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        modules, strings = references(path, files)
        for m in modules:
            f = module_file(m, files)
            if f and f not in seen:
                queue.append(f)
            elif f is None and m.split('.')[0] in PACKAGE_ROOTS and m.count('.') >= 1:
                # `from scripts import x` adds 'scripts.x' (resolved); attribute imports land here.
                parent = module_file(m.rsplit('.', 1)[0], files)
                if parent is None:
                    unresolved.add(m)
        for s in strings:
            if MODULE_STRING.match(s):
                f = module_file(s, files)
                if f and f not in seen:
                    queue.append(f)
            if SEALED.search(s):
                sealed_refs.add(s)
                continue
            if s.startswith('/') and s.split('LeWMQuad-v3/', 1)[-1] in files:
                rel = s.split('LeWMQuad-v3/', 1)[-1]
                if rel not in seen and rel.endswith(('.py', '.sh', '.json', '.yaml', '.yml')):
                    queue.append(rel)
                if not rel.endswith(('.py', '.sh')):
                    data.add(rel)
            elif s.startswith('/'):
                artefacts.add(s)
            elif PATH_STRING.match(s):
                if s in files:
                    if s.endswith(('.py', '.sh', '.json', '.yaml', '.yml')):
                        if s not in seen:
                            queue.append(s)
                    if not s.endswith(('.py', '.sh')):
                        data.add(s)
                elif '/' in s:
                    artefacts.add(s)
    return seen, data, unresolved, artefacts, sealed_refs


def main(out):
    files, sealed = tracked()
    report = dict(tracked_files=len(files), sealed_tracked_files_excluded_by_name=len(sealed), groups={})
    union, union_data, union_artefacts, union_sealed = set(), set(), set(), set()
    for group, entries in ENTRY_POINTS.items():
        missing = [e for e in entries if e not in files]
        if missing:
            raise SystemExit(f'entry points not tracked: {missing}')
        code, data, unresolved, artefacts, sealed_refs = closure(entries, files, sealed)
        report['groups'][group] = dict(entries=entries, python_files=len(code), data_files=len(data),
                                       unresolved_modules=sorted(unresolved)[:50])
        union |= code
        union_data |= data
        union_artefacts |= artefacts
        union_sealed |= sealed_refs
    by_top = defaultdict(int)
    for f in union:
        by_top[f.split('/')[0]] += 1
    report.update(python_files=sorted(union), python_file_count=len(union), python_files_by_top=dict(by_top),
                  data_files=sorted(union_data), artefact_references=sorted(union_artefacts),
                  sealed_references_excluded=sorted(union_sealed))
    text = json.dumps(report, indent=1)
    if out:
        Path(out).write_text(text+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('python_files', 'data_files', 'artefact_references')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out')
    main(p.parse_args().out)
