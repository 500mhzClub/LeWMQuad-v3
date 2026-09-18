"""Small CPU operations and caught boundary violations; no simulator or GPU work."""
import sys
import threading
from types import FunctionType

import cv2
import genesis
import pytest
import torch
from scripts.independent_round_trip_audit_cpu_monitor_development import AuditCPUMonitor


def test_actual_cpu_operations_and_opencl_state_restoration():
    before = cv2.ocl.useOpenCL(); monitor = AuditCPUMonitor()
    with monitor:
        assert not cv2.ocl.useOpenCL()
        x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        result = x @ x.T + 1
        torch.testing.assert_close(result, torch.tensor([[6., 15.], [15., 51.]]))
    summary = monitor.summary()
    assert summary['scope_returned_without_error'] and summary['torch_operation_count'] > 0
    assert not summary['violations'] and not summary['full_raw_auditor_qualified']
    assert not summary['overlap_execution_permitted'] and not summary['os_device_access_isolated']
    assert cv2.ocl.useOpenCL() == before and sys.getprofile() is None


def test_non_cpu_device_request_is_rejected_without_accelerator_initialization():
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='non-CPU device request'):
        with monitor: torch.empty(2, device='meta')
    assert not monitor.summary()['scope_returned_without_error']
    assert not torch.cuda.is_initialized()


def test_existing_non_cpu_tensor_input_is_rejected():
    x = torch.empty(2, device='meta'); monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='non-CPU tensor'):
        with monitor: x + 1


def test_genesis_call_blocked_before_its_python_body():
    entered = []
    def fake(): entered.append(True)
    # Only its module identity is synthetic; the profile call event is real.
    forbidden = FunctionType(fake.__code__, {'__name__':'genesis.synthetic', 'entered':entered},
        closure=fake.__closure__)
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='Genesis runtime call'):
        with monitor: forbidden()
    assert entered == [] and monitor.summary()['violations']


def test_real_genesis_scene_entry_is_intercepted_without_initializing_runtime():
    assert genesis._initialized is False
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='Genesis runtime call'):
        with monitor: genesis.Scene()
    assert genesis._initialized is False
    assert any('genesis.' in value for value in monitor.summary()['violations'])


def test_caught_profile_violation_still_rejects_scope():
    def fake(): pass
    forbidden = FunctionType(fake.__code__, {'__name__':'genesis.synthetic'})
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='rejected scope'):
        with monitor:
            try: forbidden()
            except ValueError: pass
    assert not monitor.summary()['scope_returned_without_error']


def test_caught_torch_violation_still_rejects_scope():
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='rejected scope'):
        with monitor:
            try: torch.empty(1, device='meta')
            except ValueError: pass
    assert monitor.summary()['violations']


def test_opencl_reenable_rejected_and_original_setting_restored():
    before = cv2.ocl.useOpenCL(); monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='OpenCL setting changed'):
        with monitor: cv2.ocl.setUseOpenCL(True)
    assert cv2.ocl.useOpenCL() == before


def test_new_python_thread_rejected_before_start():
    entered = []; thread = threading.Thread(target=lambda:entered.append(True))
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='new Python thread'):
        with monitor: thread.start()
    assert not thread.is_alive() and entered == []


def test_removed_profile_cannot_yield_success():
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='profile hook changed'):
        with monitor: sys.setprofile(None)
    assert not monitor.summary()['scope_returned_without_error']


def test_existing_profile_rejected_without_replacement():
    def profile(*args): pass
    sys.setprofile(profile)
    try:
        with pytest.raises(ValueError, match='existing profiler'): AuditCPUMonitor().__enter__()
        assert sys.getprofile() is profile
    finally: sys.setprofile(None)


def test_original_unrelated_error_propagates_and_settings_restore():
    before = cv2.ocl.useOpenCL(); monitor = AuditCPUMonitor()
    with pytest.raises(RuntimeError, match='original failure'):
        with monitor:
            torch.ones(2)
            raise RuntimeError('original failure')
    assert not monitor.summary()['scope_returned_without_error']
    assert monitor.summary()['violations'] == []
    assert sys.getprofile() is None and cv2.ocl.useOpenCL() == before


def test_monitor_is_one_use_and_summary_requires_closed_scope():
    monitor = AuditCPUMonitor()
    with pytest.raises(ValueError, match='closed monitored scope'): monitor.summary()
    with monitor: torch.zeros(1)
    with pytest.raises(ValueError, match='one-use'): monitor.__enter__()
