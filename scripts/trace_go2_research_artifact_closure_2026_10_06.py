"""Static dependency closure of the navigation-capability programme's entry points (consolidation, 6 October 2026).

For the research-artefact migration plan (docs/go2_research_artifact_migration_plan_2026-10-06.md). Starting from the
entry points listed below, it follows, in tracked repository Python files only:
- every `import` / `from ... import` statement anywhere in a file (including inside functions);
- string constants naming a repository module (`lewm.x`, `scripts.y`) or a repository file path (`scripts/z.py`,
  `config/...`, `docs/...`), which covers importlib, runpy and file-path loads.
Non-Python files reached through string constants (configs, docs, JSON bindings) are listed but not parsed.
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
PATH_STRING = re.compile(r'^[A-Za-z0-9_./-]+\.(py|json|yaml|yml|md|txt|npz|pt|csv|sha256|png|jpg|obj|urdf|xml)$')

ENTRY_POINTS = {
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


def references(path, files):
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
            if s.startswith('/'):
                artefacts.add(s)
            elif PATH_STRING.match(s):
                if s in files:
                    if s.endswith('.py'):
                        if s not in seen:
                            queue.append(s)
                    else:
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
