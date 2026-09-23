"""Single-owner bounded blinded pilot. Ends before comparative audit authority."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import tempfile
import time
import traceback

import psutil


CAPS = Path('docs/go2_decision_headroom_phase1_caps_v1_2026-09-23.json')
REPO = Path(__file__).resolve().parents[1]
GIB = 1024**3
SOURCES = (
    'scripts/run_go2_decision_headroom_pilot_development.py',
    'scripts/run_go2_decision_headroom_source_development.py',
    'scripts/run_go2_decision_headroom_branches_development.py',
    'scripts/qualify_go2_decision_headroom_reference_development.py',
    'lewm/decision_headroom_packet_development.py',
    'lewm/decision_headroom_snapshot_development.py',
    'lewm/decision_headroom_reference_development.py',
    'scripts/check_go2_decision_headroom_pilot_development.py',
    'scripts/time_go2_decision_headroom_components_development.py',
    'scripts/read_go2_decision_headroom_pilot_development.py',
    'scripts/run_go2_dense_horizon_navigation_development.py',
    'lewm/dense_horizon_navigation_development.py',
    'lewm/dense_native_observation_development.py',
    'scripts/paced_native_session_development.py',
    'scripts/whole_task_physics_session_development.py',
    'scripts/rgb_navigation_retention_development.py',
    'lewm/dense_world_model_maze_layouts_development.py',
)


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def retained_bytes(root, *, output=True):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d != 'sealed' and not d.startswith('sealed_')
                   and not (Path(directory) / d).is_symlink()]
        for name in files:
            if name == 'sealed_test.json':
                continue
            path = Path(directory) / name
            if path.is_symlink():
                if output:
                    raise ValueError('pilot output must not contain links to unrelated artifacts')
                continue
            total += path.stat().st_size
    return total


def vram():
    devices = []
    for card in Path('/sys/class/drm').glob('card[0-9]'):
        total, used = card/'device/mem_info_vram_total', card/'device/mem_info_vram_used'
        if total.is_file() and used.is_file():
            devices.append(dict(path=str(card), total=int(total.read_text()), used=int(used.read_text())))
    if not devices:
        raise RuntimeError('GPU memory measurement unavailable')
    return max(devices, key=lambda d:d['total'])


class PilotBudget:
    active = None

    def __init__(self, root, caps, admission):
        self.root, self.caps, self.admission = root, caps, admission
        self.owner = psutil.Process()
        self.started = time.monotonic()
        self.cpu_start = self._cpu_seconds()
        self.sources, self.snapshots, self.branches = set(), set(), set()
        self.components = set()
        self.source_ns = {}
        self.active_case = None
        self.stopped = False
        self.measurements = []
        self.last_check = -math.inf
        self.journal = (root/'budget_events.jsonl').open('x')
        self.peak_rss = self.peak_vram = self.peak_retained = 0
        self.cache_baseline = {p:retained_bytes(Path(p), output=False) for p in admission.get('cache_paths', [])}
        self.cache_growth = 0
        self.last_cache_check = -math.inf
        PilotBudget.active = self

    @classmethod
    def attach(cls, root, admission):
        instance = cls.active
        if instance is None or instance.root != root or admission['owner_pid'] != os.getpid():
            raise RuntimeError('pilot operation requires the current single budget owner')
        instance.check('attach', force=True)
        return instance

    def _cpu_seconds(self):
        cpu = self.owner.cpu_times()
        exited = resource.getrusage(resource.RUSAGE_CHILDREN)
        total = cpu.user + cpu.system + exited.ru_utime + exited.ru_stime
        for process in self.owner.children(recursive=True):
            try:
                cpu = process.cpu_times()
                total += cpu.user + cpu.system
            except psutil.NoSuchProcess:
                pass
        return total

    def event(self, kind, **data):
        self.journal.write(json.dumps(dict(kind=kind, elapsed_s=time.monotonic()-self.started, **data))+'\n')
        self.journal.flush()

    def check(self, stage, *, force=False):
        if self.stopped:
            raise RuntimeError('latched pilot budget stop')
        now = time.monotonic()
        if not force and now-self.last_check < .5:
            return
        rss = 0
        for process in [self.owner, *self.owner.children(recursive=True)]:
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        gpu = vram()
        written = retained_bytes(self.root)
        if now-self.last_cache_check > 15:
            self.cache_growth = sum(max(0, retained_bytes(Path(p), output=False)-before)
                for p,before in self.cache_baseline.items())
            self.last_cache_check = now
        row = dict(stage=stage, wall_s=now-self.started, cpu_s=self._cpu_seconds()-self.cpu_start,
            aggregate_rss_bytes=rss, available_ram_bytes=psutil.virtual_memory().available,
            gpu_used_bytes=gpu['used'], available_vram_bytes=gpu['total']-gpu['used'],
            retained_bytes=written, recovery_free_bytes=shutil.disk_usage(self.root).free,
            workspace_free_bytes=shutil.disk_usage(REPO).free,
            external_cache_growth_bytes=self.cache_growth,
            peak_additional_observed_bytes=written+self.cache_growth)
        limits = self.caps['compute_caps']; storage = self.caps['storage_caps']
        violated = []
        for key, limit in (('wall_s', limits['execution_wall_seconds']),
                ('cpu_s', limits['aggregate_cpu_seconds']), ('aggregate_rss_bytes', limits['aggregate_rss_bytes']),
                ('gpu_used_bytes', limits['gpu_allocated_bytes']), ('retained_bytes', storage['retained_bytes']-8*1024**2),
                ('peak_additional_observed_bytes', storage['peak_additional_bytes']-8*1024**2)):
            if row[key] > limit:
                violated.append(key)
        for key, limit in (('available_ram_bytes', limits['minimum_available_ram_bytes']),
                ('available_vram_bytes', limits['minimum_available_vram_bytes']),
                ('recovery_free_bytes', storage['recovery_filesystem_reserve_bytes']),
                ('workspace_free_bytes', storage['workspace_filesystem_reserve_bytes'])):
            if row[key] < limit:
                violated.append(key)
        self.last_check = now
        self.peak_rss = max(self.peak_rss, rss); self.peak_vram = max(self.peak_vram, gpu['used'])
        self.peak_retained = max(self.peak_retained, written)
        self.measurements.append(row)
        self.event('resource_check', **row, violated=violated)
        if violated:
            self.stopped = True
            raise RuntimeError('pilot cap reached: '+', '.join(violated))

    def admit_write(self, bytes_needed):
        self.check('before_write', force=True)
        limit = self.caps['storage_caps']
        if (retained_bytes(self.root)+bytes_needed+8*1024**2 > limit['retained_bytes']
                or shutil.disk_usage(self.root).free-bytes_needed < limit['recovery_filesystem_reserve_bytes']):
            self.stopped = True
            raise RuntimeError('pilot write would exceed retained-byte cap or disk reserve')

    def start_source(self, case):
        if type(case) is not int or case not in range(6) or case in self.sources or self.active_case is not None:
            raise ValueError('one fresh source assignment at a time; no duplicate source rollouts')
        if len(self.sources) >= self.caps['collection_caps']['source_rollouts']:
            raise RuntimeError('source-rollout cap reached')
        self.check('source_admission', force=True)
        self.sources.add(case); self.source_ns[case] = 0; self.active_case = case
        self.event('source_started', case=case)

    def reserve_source_physics(self, case, seconds):
        if case != self.active_case or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('positive physics reservation for current source required')
        ns = round(seconds*1e9)
        caps = self.caps['collection_caps']
        per_source = round((caps['source_settling_seconds_per_rollout'] + caps['source_policy_steps_per_rollout']*.02)*1e9)
        if self.source_ns[case]+ns > per_source or sum(self.source_ns.values())+ns > round(caps['source_physics_seconds_total']*1e9):
            raise RuntimeError('source physics cap reached')
        self.source_ns[case] += ns
        self.event('source_physics_reserved', case=case, ns=ns)

    def reserve_snapshot(self, case, frame):
        key = (case, frame)
        caps = self.caps['collection_caps']
        if case != self.active_case or frame not in caps['snapshot_frames'] or key in self.snapshots:
            raise ValueError('unique predeclared source snapshot required')
        if len(self.snapshots) >= caps['sampled_states_total']:
            raise RuntimeError('snapshot cap reached')
        self.snapshots.add(key); self.event('snapshot_reserved', case=case, frame=frame)

    def reserve_branch(self, identity):
        match = re.fullmatch(r'source_(\d{2})/state_(\d{4})/(source_trace_[0-2]|(?:hold|forward|left_arc|right_arc|left_turn|right_turn)/[0-2])', identity)
        if match is None:
            raise ValueError('branch identity is outside the fixed bank/repeat population')
        case, frame = map(int, match.group(1, 2))
        if case != self.active_case or (case, frame) not in self.snapshots or identity in self.branches:
            raise ValueError('branch must use a captured current-source state and fresh attempt identity')
        if len(self.branches) >= self.caps['branch_caps']['all_branch_attempts_total']:
            raise RuntimeError('branch/repeat cap reached')
        self.check('branch_admission', force=True)
        self.branches.add(identity); self.event('branch_reserved', identity=identity, physics_ns=800_000_000)

    def finish_source(self, case, error):
        self.event('source_finished', case=case, error=None if error is None else repr(error))
        self.active_case = None

    def reserve_component(self, kind, case, repeat):
        caps = self.caps['compute_caps']
        limits = dict(encoder=caps['encoder_timing_passes'], predictor=caps['predictor_timing_passes'],
            readout_old_data=caps['readout_timing_passes_per_existing_head'],
            readout_maze_data=caps['readout_timing_passes_per_existing_head'])
        identity = (kind,case,repeat)
        if kind not in limits or case != self.active_case or repeat not in (0,1) or identity in self.components:
            raise ValueError('one of the fixed component timing passes required')
        if sum(k==kind for k,_,_ in self.components) >= limits[kind]:
            raise RuntimeError('component timing cap reached')
        self.check('component_admission', force=True)
        self.components.add(identity)
        self.event('component_reserved', component=kind, case=case, repeat=repeat)

    def finish(self, error):
        report = dict(source_attempts=len(self.sources), snapshots=len(self.snapshots), branch_attempts=len(self.branches),
            component_timing_attempts=[list(x) for x in sorted(self.components)],
            reserved_source_physics_s=sum(self.source_ns.values())/1e9,
            reserved_branch_physics_s=len(self.branches)*.8,
            wall_s=time.monotonic()-self.started, cpu_s=self._cpu_seconds()-self.cpu_start,
            peak_sampled_aggregate_rss_bytes=self.peak_rss, peak_sampled_total_gpu_used_bytes=self.peak_vram,
            retained_bytes=retained_bytes(self.root), measurements=self.measurements,
            cache_baseline_bytes=self.cache_baseline,
            resource_stop_latched=self.stopped, error=None if error is None else repr(error),
            sampled_measurements_not_os_memory_limits=True,
            gpu_measurement_includes_other_device_users=True, phase2_authorized=False)
        save(self.root/'resource_result.json', report)
        self.journal.close()


def main():
    from scripts.run_go2_decision_headroom_source_development import require_stage_a_closed, run, ASSIGNMENTS
    from scripts.qualify_go2_decision_headroom_reference_development import qualify
    caps = json.loads(CAPS.read_text())
    root = Path(caps['output_root'])
    require_stage_a_closed(root)
    # Exact current paths are recorded before collection. No broad source export.
    paths = dict(repository=str(REPO.resolve()), artifacts=str(root.parent.resolve()),
        pilot_output=str(root.resolve()), temporary=str(root/'scratch/temp'),
        home_cache=str((Path.home()/'.cache').resolve()),
        vjepa_checkpoint=str(Path.home()/'.cache/vjepa2_1_vitl_dist_vitG_384.pt'),
        vjepa_source=str(Path.home()/'.cache/vjepa2-204698b45b3712590f06245fbfba32d3be539812'),
        predictor=str(REPO/'.generated/navigation_development_artifacts_v1/go2_horizon_dense_predictor_v1_attempt_001/action_final.pt'),
        readout=str(root.parent/'go2_maze_view_readout_v1_attempt_003/old_data_final.pt'),
        command_normalization=str(Path.home()/'.cache/lewm_go2_temporal_v03/proprio_v1/proprio_norm_stats.json'),
        gait_checkpoint=str(REPO/'models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt'),
        gait_config=str(REPO/'models/tier_a_go2_locomotion/20260516_contract_ppo/cfgs.pkl'),
        logs=str(root/'source_XX/worker.log'), scratch=str(root/'scratch'))
    cache_paths = [str((Path.home()/'.cache'/name).resolve()) for name in
        ('genesis','quadrants','gstaichi','triton','torch','mesa_shader_cache','mesa_shader_cache_db')]
    cache_paths.extend([str(REPO/'.generated/box_meshes'), '/tmp/torchinductor_'+Path.home().name])
    if root.exists():
        raise RuntimeError('existing pilot attempt preserved; no automatic restart')
    for directory, reserve, peak in ((root.parent, 12*GIB, 12*GIB), (REPO, 4*GIB, GIB)):
        if shutil.disk_usage(directory).free < reserve+peak:
            raise RuntimeError('pilot initial filesystem reserve plus peak writes unavailable')
    if psutil.virtual_memory().available < 32*GIB or vram()['total']-vram()['used'] < 4*GIB:
        raise RuntimeError('pilot memory admission unavailable')
    os.sched_setaffinity(0, set(caps['compute_caps']['cpu_affinity']))
    input_bindings = {}
    for name in ('vjepa_checkpoint', 'predictor', 'readout', 'command_normalization', 'gait_checkpoint', 'gait_config'):
        path = Path(paths[name]).resolve(strict=True)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        input_bindings[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=digest)
    root.mkdir()
    (root/'scratch/temp').mkdir(parents=True)
    os.environ['TMPDIR'] = str(root/'scratch/temp')
    tempfile.tempdir = str(root/'scratch/temp')
    filesystems = {}
    for name, value in paths.items():
        p = Path(value)
        while not p.exists():
            p = p.parent
        usage = shutil.disk_usage(p)
        filesystems[name] = dict(existing_parent=str(p), device_id=p.stat().st_dev,
            total_bytes=usage.total, available_bytes=usage.free)
    admission = dict(recorded_at=datetime.datetime.now().astimezone().isoformat(), owner_pid=os.getpid(),
        owner_created=psutil.Process().create_time(), caps_sha256=hashlib.sha256(CAPS.read_bytes()).hexdigest(),
        paths=paths, cache_paths=cache_paths, filesystems=filesystems, input_bindings=input_bindings,
        source_hashes={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in SOURCES},
        source_assignments=ASSIGNMENTS, source_readout='maze_view_old_data',
        restoration_tolerances=dict(position_m=.001, yaw_rad=math.radians(.1)),
        minimum_restored_state_fraction=.9, minimum_valid_states_per_source_controller=1,
        footprint_cost_not_hardware_qualified=True, new_fitting=False, comparative_audit_scoring=False,
        source_scene_and_models_unchanged=True, phase2_authorized=False)
    save(root/'pilot_execution_admission.json', admission)
    budget = PilotBudget(root, caps, admission)
    error = None
    try:
        budget.check('initial', force=True)
        qualify()
        sanity = json.loads((root/'reference_sanity_v1/result.json').read_text())
        if sanity['status'] != 'PASS':
            raise RuntimeError('reference sanity panel did not qualify; preserve result and stop')
        for case in range(len(ASSIGNMENTS)):
            run(case)
            # Missing fixed snapshots also consume their assigned population.
            # Three invalid states cannot recover 90% of the fixed 24 states.
            valid = 0
            for finished in range(case+1):
                rows = json.loads((root/f'source_{finished:02d}/branch_pilot_result.json').read_text())['states']
                valid += sum(r['restoration_passed'] for r in rows)
            impossible = (case+1)*4-valid > caps['collection_caps']['sampled_states_total']*(1-admission['minimum_restored_state_fraction'])
            if impossible:
                raise RuntimeError('fixed restoration-validity requirement cannot be reached within remaining assignments')
        save(root/'pilot_execution_complete.json', dict(status='EXECUTION_COMPLETE_REQUIRES_VALIDITY_CLOSEOUT',
            source_attempts=6, comparative_audit_scoring=False, phase2_authorized=False,
            next='Consolidate validity/repeats/resources, freeze and commit proposed protocol/config, then stop for explicit checkpoint-(a) approval.'))
    except BaseException as exc:
        error = exc
        save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc()))
    finally:
        budget.finish(error)
    if error is not None:
        raise error


if __name__ == '__main__':
    sys.modules['scripts.run_go2_decision_headroom_pilot_development'] = sys.modules[__name__]
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    main()
