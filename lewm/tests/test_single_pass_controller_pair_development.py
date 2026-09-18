from copy import deepcopy
import pytest
from scripts.benchmark_go2_single_pass_controller_pair_v1 import admit, order, measure, FRAMES


def valid():
    return dict(status='SINGLE_PASS_MAZE_CONTROLLER_REPLAY_COMPLETE',frames=1881,
        first_terminal_frame=1870,all_complete_decisions_exact=True,model_state_unchanged=True,
        original_failure_preserved=True,native_adoption=False,native_execution=False)


def test_complete_equivalence_admitted_without_changing_original_failure():
    r=valid();before=deepcopy(r);admit(r);assert r==before


@pytest.mark.parametrize('field,value',[('status','RUNNING'),('frames',1880),('first_terminal_frame',1869),
    ('all_complete_decisions_exact',False),('model_state_unchanged',False),
    ('original_failure_preserved',False),('native_adoption',True),('native_execution',True)])
def test_incomplete_or_different_full_replay_cannot_admit_benchmark(field,value):
    r=valid();r[field]=value
    with pytest.raises(ValueError):admit(r)


def test_order_balances_first_position_without_changing_frame_population():
    assert FRAMES==256
    assert sum(order(i)[0]=='original' for i in range(FRAMES))==128
    assert all(set(order(i))=={'original','single_pass'} for i in range(FRAMES))


def test_timing_includes_full_observe_and_actual_writer_but_no_input_decode():
    events=[];wall=iter([10,30,60]);cpu=iter([3,43]);result={'requested_command':[0.,0.,0.]}
    class Controller:
        def observe(self,p,d,f,*,now_ns,auxiliary_depth):
            assert (p,d,f,auxiliary_depth,now_ns)==(1,2,3,4,5)
            events.append('observe');return result
    def append(row):
        assert row==dict(tick=0,decision=result);events.append('write')
    decision,t=measure(Controller(),(1,2,3,4,5),0,append,
        clock=lambda:next(wall),cpu_clock=lambda:next(cpu))
    assert decision is result and events==['observe','write']
    assert t==dict(controller_wall_ms=20/1e6,receipt_wall_ms=30/1e6,
        controller_and_receipt_wall_ms=50/1e6,controller_and_receipt_process_cpu_ms=40/1e6)
