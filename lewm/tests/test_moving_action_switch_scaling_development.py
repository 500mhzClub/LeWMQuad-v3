import copy
from scripts.probe_go2_moving_action_switch_scaling_v1 import cases,decision


def phase(wall_s):
    return dict(wall_s=wall_s,all_workers_completed=True,all_measurement_gates_pass=True,
        records={t:dict(status='MOVING_ACTION_SWITCH_COLLECTED_AND_AUDITED',hard_measurement_failed_frames=[],
            strict_physical_visibility_pass=True,outcome=dict(complete_schedule=True),
            collection=dict(acquisition_stop=None,command_ticks=63,rgbd_frames=64,physics_samples=3900)) for t in cases()})


def test_parallel_selection_needs_exact_outputs_full_schedule_and_speedup():
    serial=phase(100);parallel=phase(50);comparisons=[dict(trial=t,exact_equal=True) for t in cases()]
    assert decision(serial,parallel,comparisons)['selected_workers']==4
    comparisons[0]['exact_equal']=False
    assert decision(serial,parallel,comparisons)['selected_workers']==1
    comparisons[0]['exact_equal']=True;parallel['wall_s']=90
    assert decision(serial,parallel,comparisons)['selected_workers']==1
    parallel['wall_s']=50;parallel['records'][cases()[0]]['collection']['rgbd_frames']=63
    assert decision(serial,parallel,comparisons)['selected_workers']==1
    serial['records'][cases()[0]]['strict_physical_visibility_pass']=False
    assert decision(serial,parallel,comparisons)['selected_workers'] is None
