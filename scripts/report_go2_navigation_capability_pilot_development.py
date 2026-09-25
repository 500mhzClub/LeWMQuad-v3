"""Summarise completed pilots and prospective workload; no new physics or scoring."""
import json
import os
from pathlib import Path
import shutil
import time

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import PROTOCOL, REPO, save, sha


def storage(path):
    logical = allocated = 0
    for directory, directories, files in os.walk(path):
        directories[:] = [n for n in directories if n != 'sealed' and not n.startswith('sealed_')]
        for name in files:
            if name == 'sealed_test.json':
                continue
            s = (Path(directory)/name).stat()
            logical += s.st_size
            allocated += s.st_blocks*512
    return dict(logical_bytes=logical, allocated_bytes=allocated)


def main():
    protocol = json.loads(PROTOCOL.read_text()); base = Path(protocol['output_root'])
    output.install(base)
    pilots = {}
    for arm in ['C0', 'C1', 'C2', 'C3', 'C4']:
        root = base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm == "C0" else 1:03d}'
        result = json.loads((root/'result.json').read_text())
        evaluation = json.loads((root/'episode_evaluation.json').read_text())
        pilots[arm] = dict(root=str(root), result=result,
            evaluation={k:v for k,v in evaluation.items() if k not in ['hold_details', 'original_arrival_reader', 'arrivals']},
            storage=storage(root), bindings={n:sha(root/n) for n in ['config.json', 'result.json', 'episode_evaluation.json']})
    concurrency_path = base/'concurrency_C3_v0_attempt001/result.json'
    concurrency = json.loads(concurrency_path.read_text())
    c2_check = json.loads((base/'equivalence/C2_unused_workload_serial_attempt001/result.json').read_text())
    c1_video = json.loads((base/'videos/pipeline_test_attempt001/metadata.json').read_text())
    chase = json.loads((base/'videos/pipeline_test_chase_revision002/metadata.json').read_text())
    ratios = {a:p['result']['wall_s']/p['result']['simulated_s'] for a,p in pilots.items()}
    # The whole C1 replay-plus-video measurement conservatively bounds its
    # qualified unused-workload omission; no unmeasured speedup is credited.
    ratios['C1'] = c1_video['wall_s']/pilots['C1']['result']['simulated_s']
    ratios['C2'] = c2_check['wall_s']/c2_check['simulated_s']
    serial_ratios = dict(ratios)
    admitted = [l for l in concurrency['levels'] if l['passed']]
    if concurrency['passed'] and admitted:
        ratios['C3'] = min(ratios['C3'], min(l['wall_s']/l['aggregate_simulated_s'] for l in admitted))
    native_ratios = {a:p['evaluation']['articulated_reader_wall_s']/p['result']['simulated_s'] for a,p in pilots.items()}
    storage_rates = {a:p['storage']['allocated_bytes']/p['result']['simulated_s'] for a,p in pilots.items()}
    # C0's initial camera omission must not understate future full recordings.
    storage_rates['C0'] = max(storage_rates['C0'], storage_rates['C1'])
    origin = json.loads((base/'wall_budget_origin.json').read_text())['started_unix_s']
    elapsed_h = (time.time()-origin)/3600
    video_ratio = chase['wall_s']/pilots['C1']['result']['simulated_s']
    scenarios = {}
    for episodes_per_maze in [2, 1]:
        validation_per_arm = 20*episodes_per_maze
        # Reuse the completed v0 C1 screen episode and C0 gate episode.
        counts = dict(C0=19+10, C1=9+validation_per_arm,
                      C2=validation_per_arm, C3=validation_per_arm, C4=validation_per_arm)
        # All future missions consume 480 s in this planning scenario. This is
        # a workload assumption, not an estimated distribution across mazes.
        run_h = sum(counts[a]*480*(ratios[a]+native_ratios[a]) for a in counts)/3600
        # Four full decision replays, four chase passes; reserve a second
        # chase-pass cost for encoding cuts/composites/contact sheets.
        video_h = 480*(sum(serial_ratios[a] for a in ['C1','C2','C3','C4'])+8*video_ratio)/3600
        projected = elapsed_h+1.15*(run_h+video_h)
        retained = sum(counts[a]*480*storage_rates[a] for a in counts)
        video_storage = storage(base/'videos/pipeline_test_chase_revision002')['allocated_bytes']
        retained += 8*480*video_storage/pilots['C1']['result']['simulated_s']
        # Also show a shorter, pilot-duration workload to expose dependence
        # on the full-mission assumption. Neither is a storage guarantee.
        pilot_length = sum(counts[a]*pilots[a]['result']['simulated_s']*storage_rates[a] for a in counts)
        scenarios[str(episodes_per_maze)] = dict(validation_episodes_per_maze=episodes_per_maze,
            remaining_science_episodes=counts, remaining_run_and_native_analysis_h=run_h,
            remaining_video_h=video_h, time_contingency_fraction=.15,
            projected_total_wall_h=projected, within_120h_projection=projected <= 120,
            full_480s_recording_and_video_bytes=retained,
            pilot_length_science_recording_bytes=pilot_length)
    chosen = 2 if scenarios['2']['within_120h_projection'] else 1
    free = shutil.disk_usage(base).free
    available = free-protocol['caps']['recovery_reserve_bytes']-128*1024**2
    report = dict(schema='navigation_capability_pilot_budget.v1', pilots=pilots,
        concurrency=concurrency, effective_execution_wall_per_sim_s=ratios,
        native_reader_wall_per_sim_s=native_ratios, allocated_storage_bytes_per_sim_s=storage_rates,
        programme_elapsed_h=elapsed_h, scenarios=scenarios,
        time_rule_selected_validation_episodes_per_maze=chosen,
        time_rule_stop=not scenarios[str(chosen)]['within_120h_projection'],
        recovery_free_bytes=free, recovery_available_after_reserve_and_closeout_bytes=available,
        workspace_free_bytes=shutil.disk_usage(REPO).free,
        next_source_existing_3GiB_admission_passes=available >= 3*1024**3,
        projected_storage_fits=scenarios[str(chosen)]['full_480s_recording_and_video_bytes'] <= available,
        retained_existing_artifacts_unchanged=True,
        limitations=['One paired development episode per controller; no population capability inference.',
            '480-s future missions and 15% timing contingency are explicit planning assumptions, not statistical confidence bounds.',
            'Projection covers the remaining v0 screen, one oracle gate and qualification; additional harness versions require renewed remaining-budget accounting.',
            'No C0 concurrency or C1/C2/C4 concurrency speedup is assumed.',
            'Only a fully passing concurrency package admits the C3 speedup.',
            'Training-render provenance remains unverified.'],
        bindings=dict(preregistration=sha(PROTOCOL), script=sha(__file__), concurrency=sha(concurrency_path)))
    root = base/'pilot_budget_report_attempt001'; root.mkdir(exist_ok=False)
    save(root/'result.json',report)
    print(output.dumps({k:v for k,v in report.items() if k not in ['pilots','concurrency','bindings']}))


if __name__ == '__main__':
    main()
