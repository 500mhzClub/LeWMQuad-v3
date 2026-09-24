"""Small synthetic checks only: no physics, models, or scientific scoring."""
import json
from pathlib import Path
import tempfile
import time
import unittest
import os
import subprocess
import sys
from unittest.mock import patch

from lewm.decision_headroom_resume_monitor_development import ClosedAssignmentSizes, ResumeBudget
from scripts.handover_go2_headroom_v42_development import reconstruct
from scripts.run_go2_decision_headroom_pilot_development import retained_bytes
import psutil


def event(root, **value):
    with (root / 'budget_events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(elapsed_s=1., **value))+'\n')


def closed_fixture(root):
    (root / 'source_00').mkdir()
    (root / 'source_00' / 'snapshots.json').write_text('[{"frame":12}]')
    event(root, kind='source_started', case=0)
    event(root, kind='source_physics_reserved', case=0, ns=1500000000)
    event(root, kind='branch_reserved', identity='source_00/state_0012/hold_0')
    event(root, kind='resource_check', violated=[], cpu_s=3., aggregate_rss_bytes=4,
          gpu_used_bytes=5, retained_bytes=6, external_cache_growth_bytes=0,
          wall_s=1.)
    event(root, kind='source_finished', case=0)
    (root / 'cell_00_closeout.json').write_text('{"case":0,"status":"complete"}')


class HandoverChecks(unittest.TestCase):
    def test_live_boundary_watcher_on_synthetic_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed_fixture(root)
            (root / 'cell_00_closeout.json').unlink()
            (root / 'pilot_execution_admission.json').write_text('{}')
            child_code = '''import json,sys,time
from pathlib import Path
r=Path(sys.argv[1])
while not (r/'trigger').exists(): time.sleep(.01)
(r/'cell_00_closeout.json').write_text(json.dumps(dict(case=0,status='complete')))
time.sleep(3)
(r/'would_start_next').write_text('unexpected')
time.sleep(30)
'''
            child = subprocess.Popen([sys.executable, '-c', child_code, str(root)])
            watcher = None
            try:
                watcher_code = '''import sys
from scripts.handover_go2_headroom_v42_development import wait_boundary
wait_boundary(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]),dict(execution_order=[0]))
'''
                watcher = subprocess.Popen([sys.executable, '-c', watcher_code, str(root), str(child.pid),
                    str(psutil.Process(child.pid).create_time())], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertIn('WAITING_FOR_COMPLETE_ASSIGNMENT', watcher.stdout.readline())
                (root / 'trigger').touch()
                stdout, stderr = watcher.communicate(timeout=15)
                self.assertEqual(watcher.returncode, 0, stderr)
                self.assertIn('SAFE_BOUNDARY_PAUSED', stdout)
                self.assertEqual(psutil.Process(child.pid).status(), psutil.STATUS_STOPPED)
                self.assertFalse((root / 'would_start_next').exists())
                boundary = json.loads((root / 'handover_boundary.json').read_text())
                self.assertEqual(boundary['closed_cases'], [0])
                self.assertEqual(boundary['remaining_cases'], [1, 2])
            finally:
                child.kill()
                child.wait()
                if watcher is not None and watcher.poll() is None:
                    watcher.kill()
                    watcher.wait()

    def test_boundary_and_next_admission_race(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed_fixture(root)
            state = reconstruct(root, [0, 1, 2])
            self.assertEqual(state['remaining_cases'], [1, 2])
            self.assertEqual(state['source_ns'], {0: 1500000000})
            self.assertEqual(state['branches'], ['source_00/state_0012/hold_0'])
            event(root, kind='source_started', case=1)
            self.assertIsNone(reconstruct(root, [0, 1, 2]))

    def test_failed_assignment_is_preserved_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed_fixture(root)
            (root / 'cell_00_closeout.json').write_text('{"case":0,"status":"unresolved","reason":"fixture"}')
            state = reconstruct(root, [0, 1])
            self.assertEqual(state['outcomes'][0]['status'], 'unresolved')
            self.assertEqual(state['remaining_cases'], [1])

    def test_accounting_and_reconciliation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed_fixture(root)
            sizes = ClosedAssignmentSizes(root, [0])
            active = root / 'source_01'
            active.mkdir()
            for n in (10, 200, 7):
                (active / 'growing.bin').write_bytes(b'x'*n)
                (root / 'root_output.bin').write_bytes(b'y'*(2*n))
                self.assertEqual(sizes.current(), retained_bytes(root))
            sizes.seal(1)
            self.assertEqual(sizes.reconcile(), retained_bytes(root))
            (root / 'source_00' / 'unexpected.bin').write_bytes(b'x')
            with self.assertRaisesRegex(RuntimeError, 'closed assignment size changed'):
                sizes.reconcile()

    def test_cumulative_budget_and_no_cap_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed_fixture(root)
            boundary = reconstruct(root, [0, 1])
            boundary.update(original_wall_origin_epoch=time.time()-100., prior_cpu_s=50.)
            gib = 1024**3
            caps = dict(compute_caps=dict(execution_wall_seconds=200., aggregate_cpu_seconds=1000.,
                aggregate_rss_bytes=100*gib, gpu_allocated_bytes=8*gib,
                minimum_available_ram_bytes=0, minimum_available_vram_bytes=0,
                gpu_owner_wall_seconds=200.), storage_caps=dict(retained_bytes=gib,
                peak_additional_bytes=2*gib, recovery_filesystem_reserve_bytes=0,
                workspace_filesystem_reserve_bytes=0))
            meter = ResumeBudget(root, caps, dict(cache_paths=[]), boundary)
            try:
                with patch('lewm.decision_headroom_resume_monitor_development.vram', return_value=dict(used=0, total=20*gib)):
                    meter.check('fixture', force=True)
                    self.assertGreaterEqual(meter.measurements[-1]['wall_s'], 100.)
                    self.assertGreaterEqual(meter.measurements[-1]['cpu_s'], 50.)
                    self.assertEqual(meter.sources, {0})
                    self.assertEqual(len(meter.branches), 1)
                    with self.assertRaises(ValueError):
                        meter.start_source(0)
                    with self.assertRaisesRegex(RuntimeError, 'write exceeds'):
                        meter.admit_write(2*gib)
                    self.assertTrue(meter.stopped)
            finally:
                meter.journal.close()

    def test_original_wall_cap_and_gpu_cap_still_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            closed_fixture(root)
            boundary = reconstruct(root, [0, 1])
            boundary.update(original_wall_origin_epoch=time.time()-201., prior_cpu_s=50.)
            gib = 1024**3
            caps = dict(compute_caps=dict(execution_wall_seconds=200., aggregate_cpu_seconds=1000.,
                aggregate_rss_bytes=100*gib, gpu_allocated_bytes=8*gib,
                minimum_available_ram_bytes=0, minimum_available_vram_bytes=0,
                gpu_owner_wall_seconds=200.), storage_caps=dict(retained_bytes=gib,
                peak_additional_bytes=2*gib, recovery_filesystem_reserve_bytes=0,
                workspace_filesystem_reserve_bytes=0))
            meter = ResumeBudget(root, caps, dict(cache_paths=[]), boundary)
            try:
                with patch('lewm.decision_headroom_resume_monitor_development.vram', return_value=dict(used=0, total=20*gib)):
                    with self.assertRaisesRegex(RuntimeError, 'wall_s, gpu_owner_wall_s'):
                        meter.check('fixture', force=True)
            finally:
                meter.journal.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
