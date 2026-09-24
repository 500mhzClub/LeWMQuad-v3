"""Instrument a reviewed single-thread Python audit scope for CPU-only checks.

This is runtime instrumentation, not an OS sandbox or full-auditor qualification.
It must wrap the actual audit before its measurements can support overlap.
"""
from collections import Counter
import sys
import threading

import cv2
import genesis
import torch
from torch.utils._python_dispatch import TorchDispatchMode

SOURCE = 'scripts/independent_round_trip_audit_cpu_monitor_development.py'
TEST = 'lewm/tests/test_independent_round_trip_audit_cpu_monitor_development.py'
PROTOCOL = 'docs/go2_independent_round_trip_audit_cpu_monitor_v1_2026-09-11.md'


class _CPUTensors(TorchDispatchMode):
    def __init__(self, monitor):
        super().__init__(); self.monitor = monitor

    def check(self, value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu': self.monitor.reject('non-CPU tensor: '+str(value.device))
        elif isinstance(value, torch.device):
            if value.type != 'cpu': self.monitor.reject('non-CPU device request: '+str(value))
        elif isinstance(value, dict):
            for item in value.values(): self.check(item)
        elif isinstance(value, (list, tuple)):
            for item in value: self.check(item)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        self.check(args); self.check(kwargs)
        result = func(*args, **kwargs)
        self.check(result)
        self.monitor.operations[str(func)] += 1
        return result


class AuditCPUMonitor:
    """One-use monitor for the actual scope; violations survive caught errors."""
    def __init__(self):
        self.used = False; self.closed = False; self.closing = False
        self.violations = []; self.operations = Counter(); self.python_calls = 0; self.c_calls = 0
        self.scope_returned_without_error = False
        self.callback = self.profile
        self.opencl_setter = cv2.ocl.setUseOpenCL
        self.mode = _CPUTensors(self)

    def reject(self, message):
        self.violations.append(message)
        raise ValueError('audit CPU monitor: '+message)

    def profile(self, frame, event, arg):
        if self.closing: return
        if event == 'call':
            self.python_calls += 1
            module = frame.f_globals.get('__name__', '')
            name = frame.f_code.co_name
            if module == 'genesis' or module.startswith('genesis.'):
                self.reject('Genesis runtime call: '+module+'.'+name)
            if module in ('torch.cuda', 'torch.xpu') and name == '_lazy_init':
                self.reject('accelerator initialization: '+module+'.'+name)
            if frame.f_code is threading.Thread.start.__code__:
                self.reject('new Python thread outside single-thread audit scope')
        elif event == 'c_call':
            self.c_calls += 1
            module = getattr(arg, '__module__', '') or ''
            name = getattr(arg, '__name__', '')
            if module == 'genesis' or module.startswith('genesis.'):
                self.reject('Genesis native call: '+module+'.'+name)
            if module == 'torch._C' and name in ('_cuda_init', '_xpu_init'):
                self.reject('native accelerator initialization: '+name)
            if arg is self.opencl_setter:
                self.reject('OpenCL setting changed inside audited scope')
            if module == '_thread' and name in ('start_new_thread', 'start_new'):
                self.reject('new Python thread outside single-thread audit scope')

    def __enter__(self):
        if self.used: raise ValueError('fresh one-use audit CPU monitor required')
        self.used = True
        if sys.getprofile() is not None or threading.active_count() != 1:
            raise ValueError('single Python thread without an existing profiler required')
        if torch.cuda.is_initialized() or torch.get_default_device().type != 'cpu':
            raise ValueError('uninitialized accelerator and CPU default device required')
        if getattr(genesis, '_initialized', None) is not False:
            raise ValueError('fresh process without initialized Genesis required')
        self.previous_opencl = cv2.ocl.useOpenCL()
        self.opencl_setter(False)
        if cv2.ocl.useOpenCL(): raise ValueError('OpenCV OpenCL could not be disabled')
        self.mode.__enter__()
        sys.setprofile(self.callback)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if sys.getprofile() is not self.callback:
                self.violations.append('Python profile hook changed or was removed')
            if threading.active_count() != 1:
                self.violations.append('unexpected Python threads at scope exit')
            if torch.cuda.is_initialized() or torch.get_default_device().type != 'cpu':
                self.violations.append('accelerator initialized or default device changed')
            if getattr(genesis, '_initialized', None) is not False:
                self.violations.append('Genesis initialized during scope')
            if cv2.ocl.useOpenCL():
                self.violations.append('OpenCL enabled during scope')
        finally:
            self.closing = True
            sys.setprofile(None)
            self.mode.__exit__(exc_type, exc, tb)
            self.opencl_setter(self.previous_opencl)
            self.closed = True
        self.scope_returned_without_error = exc_type is None and not self.violations
        if self.violations:
            raise ValueError('audit CPU monitor rejected scope: '+ '; '.join(self.violations)) from exc
        return False

    def summary(self):
        if not self.closed: raise ValueError('closed monitored scope required')
        return dict(scope_returned_without_error=self.scope_returned_without_error,
            violations=list(self.violations), torch_operations=dict(sorted(self.operations.items())),
            torch_operation_count=sum(self.operations.values()), python_calls=self.python_calls,
            native_python_calls=self.c_calls, initial_opencl_enabled=self.previous_opencl,
            opencl_disabled_inside_scope=True, original_opencl_setting_restored=True,
            tensor_inputs_outputs_and_device_requests_checked=True,
            genesis_runtime_calls_prohibited=True, accelerator_initialization_prohibited=True,
            new_python_threads_prohibited=True, os_device_access_isolated=False,
            full_raw_auditor_qualified=False, overlap_execution_permitted=False)
