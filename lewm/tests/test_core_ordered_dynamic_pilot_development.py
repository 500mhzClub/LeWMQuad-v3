import ast
from pathlib import Path
from types import SimpleNamespace as NS
import pytest

def function(path,name):
    tree=ast.parse(Path(path).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)

def test_collect_science_identical_except_session_symbol():
    old=function('scripts/run_go2_ordered_union_dynamic_sensor_pilot_v1.py','collect')
    new=function('scripts/run_go2_core_ordered_union_dynamic_sensor_pilot_v1.py','collect')
    for n in ast.walk(new):
        if isinstance(n,ast.Name) and n.id=='CoreOrderedDynamicSession':n.id='OrderedDynamicSession'
    assert ast.dump(old)==ast.dump(new)

def test_terminal_audit_computation_unchanged():
    for name in ('comparisons','load_terminal','main'):
        old=function('scripts/audit_go2_ordered_union_dynamic_sensor_pilot_v1.py',name)
        new=function('scripts/audit_go2_core_ordered_union_dynamic_sensor_pilot_v1.py',name)
        assert ast.dump(old)==ast.dump(new)

def test_only_capture_override_and_distinct_output_protocol():
    from scripts.core_ordered_dynamic_session_development import CoreOrderedDynamicSession as New
    from scripts.ordered_dynamic_session_development import OrderedDynamicSession as Old
    from scripts.run_go2_core_ordered_union_dynamic_sensor_pilot_v1 import OUTPUT,PROTOCOL,RUNS
    from scripts.run_go2_ordered_union_dynamic_sensor_pilot_v1 import OUTPUT as old_output,PROTOCOL as old_protocol,RUNS as old_runs
    assert RUNS==old_runs and OUTPUT!=old_output and PROTOCOL!=old_protocol
    assert New.__init__ is Old.__init__ and New._sample is Old._sample and New.command_tick is Old.command_tick
    a=function('scripts/ordered_dynamic_session_development.py','capture_fixed_rgb')
    b=function('scripts/core_ordered_dynamic_session_development.py','capture_fixed_rgb')
    assert ast.dump(a)==ast.dump(b)

def test_actual_override_calls_corrected_query(monkeypatch,tmp_path):
    import scripts.core_ordered_dynamic_session_development as mod
    row={'rgb_sha256':'unchanged'};calls=[]
    monkeypatch.setattr(mod.NearFieldCapture,'capture_fixed_rgb',lambda *a:row)
    monkeypatch.setattr(mod,'verify_order',lambda c,w:w)
    monkeypatch.setattr(mod,'precision_readback',lambda c:calls.append(c) or {'depth_target_depth_bits':24})
    camera=object();s=NS(ctx=NS(build=NS(camera=camera)),samples=[None]*750,raster_order={'order':'floor_first'})
    assert mod.CoreOrderedDynamicSession.capture_fixed_rgb(s,tmp_path,'rgb_0000') is row and calls==[camera]
    with pytest.raises(FileExistsError):mod.CoreOrderedDynamicSession.capture_fixed_rgb(s,tmp_path,'rgb_0000')
