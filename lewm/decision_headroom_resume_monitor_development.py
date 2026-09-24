"""Monitoring-only continuation of V4.2, with immutable closed-cell size caches."""
import json
import math
import shutil
import time
from pathlib import Path

import psutil

from scripts.run_go2_decision_headroom_pilot_development import PilotBudget, REPO, retained_bytes, vram
from scripts.run_go2_headroom_v42_development import AuditBudget


class ClosedAssignmentSizes:
    """Scan the writable assignment, not every completed assignment, each check.

    Full reconciliation at every assignment boundary and final closeout detects
    unexpected growth of a closed tree. The frozen scientific writers only write
    their own assignment; analysis reads those trees and writes root-level files.
    """
    def __init__(self, root, closed_cases):
        self.root = Path(root)
        self.closed = {}
        for case in closed_cases:
            self.seal(case)

    def seal(self, case):
        name = f'source_{case:02d}'
        self.closed[name] = retained_bytes(self.root / name)

    def current(self):
        total = 0
        for path in self.root.iterdir():
            if path.name == 'sealed' or path.name.startswith('sealed_') or path.name == 'sealed_test.json':
                continue
            if path.is_symlink():
                raise ValueError('audit output contains a symlink')
            if path.name in self.closed:
                if not path.is_dir():
                    raise ValueError('closed source disappeared')
                total += self.closed[path.name]
            elif path.is_dir():
                total += retained_bytes(path)
            else:
                total += path.stat().st_size
        if any(not (self.root / name).is_dir() for name in self.closed):
            raise ValueError('closed source disappeared')
        return total

    def reconcile(self):
        for name, expected in self.closed.items():
            if retained_bytes(self.root / name) != expected:
                raise RuntimeError('closed assignment size changed: ' + name)
        return self.current()


class ResumeBudget(AuditBudget):
    def __init__(self, root, caps, admission, boundary):
        self.root, self.caps, self.admission = root, caps, admission
        self.owner = psutil.Process()
        # Charge all elapsed time since original process creation, including the
        # handover pause. GPU owner time uses the same conservative upper bound.
        self.started = time.monotonic() - (time.time() - boundary['original_wall_origin_epoch'])
        # Include successor startup and verification CPU, not just post-admission
        # work. The predecessor carry already includes its process startup.
        self.cpu_start = -boundary['prior_cpu_s']
        self.sources = set(boundary['closed_cases'])
        self.snapshots = {tuple(s) for s in boundary['snapshots']}
        self.branches = set(boundary['branches'])
        self.source_ns = {int(k): v for k, v in boundary['source_ns'].items()}
        self.components = set()
        self.active_case = None
        self.stopped = False
        self.measurements = []
        with (root / 'budget_events.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'resource_check':
                    self.measurements.append({k: v for k, v in row.items() if k not in ('kind', 'elapsed_s', 'violated')})
        self.last_check = self.last_cache_check = -math.inf
        self.journal = (root / 'budget_events.jsonl').open('a')
        self.peak_rss = boundary['peak_rss']
        self.peak_vram = boundary['peak_vram']
        self.peak_retained = boundary['peak_retained']
        self.cache_baseline = {p: retained_bytes(Path(p), output=False) for p in admission.get('cache_paths', [])}
        self.cache_growth_carry = boundary['cache_growth_carry']
        self.cache_growth = self.cache_growth_carry
        self.sizes = ClosedAssignmentSizes(root, boundary['closed_cases'])
        self.gpu_owner_started = None
        self.gpu_owner_seconds = time.monotonic() - self.started
        self.implementation_only = False
        PilotBudget.active = self

    def check(self, stage, *, force=False):
        if self.stopped:
            raise RuntimeError('latched audit resource stop')
        now = time.monotonic()
        if not force and now - self.last_check < .5:
            return
        rss = 0
        for process in [self.owner, *self.owner.children(recursive=True)]:
            try:
                rss += process.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        gpu = vram()
        written = self.sizes.current()
        if now - self.last_cache_check > 15:
            self.cache_growth = self.cache_growth_carry + sum(
                max(0, retained_bytes(Path(p), output=False) - before)
                for p, before in self.cache_baseline.items())
            self.last_cache_check = now
        row = dict(stage=stage, wall_s=now-self.started, cpu_s=self._cpu_seconds()-self.cpu_start,
                   aggregate_rss_bytes=rss, available_ram_bytes=psutil.virtual_memory().available,
                   gpu_used_bytes=gpu['used'], available_vram_bytes=gpu['total']-gpu['used'],
                   retained_bytes=written, recovery_free_bytes=shutil.disk_usage(self.root).free,
                   workspace_free_bytes=shutil.disk_usage(REPO).free,
                   external_cache_growth_bytes=self.cache_growth,
                   peak_additional_observed_bytes=written+self.cache_growth)
        limits, storage = self.caps['compute_caps'], self.caps['storage_caps']
        violated = []
        for key, limit in (
            ('wall_s', limits['execution_wall_seconds']),
            ('cpu_s', limits['aggregate_cpu_seconds']),
            ('aggregate_rss_bytes', limits['aggregate_rss_bytes']),
            ('gpu_used_bytes', limits['gpu_allocated_bytes']),
            ('retained_bytes', storage['retained_bytes']-8*1024**2),
            ('peak_additional_observed_bytes', storage['peak_additional_bytes']-8*1024**2),
        ):
            if row[key] > limit:
                violated.append(key)
        for key, limit in (
            ('available_ram_bytes', limits['minimum_available_ram_bytes']),
            ('available_vram_bytes', limits['minimum_available_vram_bytes']),
            ('recovery_free_bytes', storage['recovery_filesystem_reserve_bytes']),
            ('workspace_free_bytes', storage['workspace_filesystem_reserve_bytes']),
        ):
            if row[key] < limit:
                violated.append(key)
        self.gpu_owner_seconds = now - self.started
        if self.gpu_owner_seconds > limits['gpu_owner_wall_seconds']:
            violated.append('gpu_owner_wall_s')
        self.last_check = now
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_vram = max(self.peak_vram, gpu['used'])
        self.peak_retained = max(self.peak_retained, written)
        self.measurements.append(row)
        self.event('resource_check', **row, violated=violated)
        if violated:
            self.stopped = True
            raise RuntimeError('audit cap reached: ' + ', '.join(violated))

    def admit_write(self, bytes_needed):
        self.check('before_write', force=True)
        storage = self.caps['storage_caps']
        # One fresh measurement per admission; no second identical tree scan.
        if (self.measurements[-1]['retained_bytes'] + bytes_needed + 8*1024**2 > storage['retained_bytes']
                or shutil.disk_usage(self.root).free - bytes_needed < storage['recovery_filesystem_reserve_bytes']):
            self.stopped = True
            raise RuntimeError('audit write exceeds retained cap or reserve')

    def close_assignment(self, case):
        self.sizes.reconcile()
        self.sizes.seal(case)
        self.check('assignment_closeout', force=True)

    def finish(self, error):
        self.sizes.reconcile()
        super().finish(error)
