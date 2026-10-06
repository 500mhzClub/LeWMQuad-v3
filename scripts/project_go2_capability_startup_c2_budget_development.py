"""Cohort budget projection from measured complete missions, with 15% reserve."""
import argparse
import json
from pathlib import Path
import shutil
import time
from scripts import run_go2_navigation_capability_correctness_c2_development as owner


def project(completed=()):
    p=json.loads(owner.PROTOCOL.read_text());base=Path(p['output_root'])
    previous=json.loads((owner.REPO/'docs/go2_navigation_capability_resume_budget_2026-09-26.json').read_text())
    counts=dict(C0=30,C1=30,C2=20,C3=20,C4=20)
    measured={arm:[] for arm in counts};readers={arm:[] for arm in counts}
    old=json.loads((base/'cohorts/v0_task_c1_C1_screen/result.json').read_text())
    previous_runs=[base/'runs'/r['assignment'] for r in old['rows']]
    for run in [*previous_runs,*completed]:
        config=json.loads((run/'config.json').read_text());arm=config['controller']
        result=json.loads((run/'result.json').read_text())
        evaluation=json.loads((run/'episode_evaluation.json').read_text())
        measured[arm].append(result['wall_s']);readers[arm].append(evaluation.get('articulated_reader_wall_s',0))
        if run in completed:counts[arm]-=1
    basis={};wall={};physical={}
    for arm in counts:
        pilot=base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm=="C0" else 1:03d}'
        result=json.loads((pilot/'result.json').read_text())
        evaluation=json.loads((pilot/'episode_evaluation_reference_v3.json').read_text())
        wall[arm]=sum(measured[arm])/len(measured[arm]) if measured[arm] else result['wall_s']
        physical[arm]=sum(readers[arm])/len(readers[arm]) if readers[arm] else evaluation['articulated_reader_wall_s']
        basis[arm]=dict(missions=len(measured[arm]) or 1,
            source='corrected completed missions including failures' if measured[arm] else 'pre-fix pilot; provisional',
            mean_wall_s_per_mission=wall[arm],mean_reader_wall_s_per_mission=physical[arm])
    origin=json.loads((base/'wall_budget_origin.json').read_text())['started_unix_s']
    elapsed=(time.time()-origin)/3600
    # Until measured on C2, conservatively charge C1 2.56x mapping-grid growth
    # against its entire mission wall cost; physical reader cost is unchanged.
    if not any(json.loads((r/'config.json').read_text())['controller']=='C1' for r in completed):
        wall['C1']*=2.56
        basis['C1']['unmeasured_grid_growth_wall_multiplier']=2.56
    future=sum(counts[a]*(wall[a]+physical[a]) for a in counts)/3600
    videos=previous['video_hours'];extra=previous['extra_replay_and_capture_allowance_hours']
    total=elapsed+1.15*(future+videos+extra)
    # Storage admission remains conservative at the full 480-s mission cap.
    logs=sum(counts[a]*480*(previous['non_image_log_allocated_bytes_per_sim_s'][a]+
        previous['extra_hash_metadata_bytes_per_sim_s']) for a in counts)
    first=base/'runs'/p['first_corrected_C0_assignment']
    full=0 if first.exists() else 480*(previous['full_retention_rgb_bytes_per_sim_s']+previous['full_retention_depth_bytes_per_sim_s'])
    storage=1.15*(logs+full+previous['video_storage_allowance_bytes'])
    usable=shutil.disk_usage(base).free-p['caps']['recovery_reserve_bytes']-128*1024**2
    return dict(schema='navigation_capability_measured_mission_budget.v1',elapsed_conservative_wall_hours=elapsed,
        remaining_science_episodes=counts,measurement_basis=basis,future_runs_and_native_analysis_hours=future,
        video_hours=videos,extra_replay_and_capture_allowance_hours=extra,contingency_fraction=.15,
        projected_total_wall_hours=total,wall_cap_hours=160,time_fits=total<=160,
        additional_storage_with_contingency_bytes=storage,recovery_usable_bytes=usable,storage_fits=storage<=usable,
        validation_episodes_per_maze=1,extra_harness_versions_assumed=0,
        assumptions=['Re-project after each cohort; future missions use measured wall time per controller, including failures.',
        'A single remaining screen/gate cycle; another outcome-driven version needs a fresh projection.',
        'No concurrency speedup credited. C1/C2 pre-fix model-workload cost is conservative until corrected measurements.',
        'C0 full recording only for first corrected gate episode; bitwise replay is a strict gate.',
        'Storage is sized at 480 s per remaining mission; time is not.'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from lewm import decision_headroom_json_v42_development as output
    output.install(a.output.parent)
    r=project();owner.save(a.output,r)
    print(json.dumps(r))
    if not r['time_fits'] or not r['storage_fits']:raise SystemExit('Budget stop')
