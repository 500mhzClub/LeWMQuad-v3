"""Measured conditional budget; no launch authority independent of the brief."""
import json
import os
from pathlib import Path
import shutil
import time
from lewm import decision_headroom_json_v42_development as output


def main():
    repo=Path(__file__).resolve().parents[1]
    protocol=json.loads((repo/'docs/go2_navigation_capability_preregistration_v1_2026-09-25.json').read_text())
    base=Path(protocol['output_root']);output.install(base)
    old=json.loads((repo/'docs/go2_navigation_capability_pilot_budget_2026-09-25.json').read_text())
    replays=base/'sensor_regeneration_2026-09-26'
    results={a:json.loads((replays/a/'result.json').read_text()) for a in ('C0','C1','C2','C3','C4')}
    measurement=json.loads((replays/'C4/native_depth_storage_measurement.json').read_text())
    remaining=dict(C0=30,C1=30,C2=20,C3=20,C4=20)
    rates=old['effective_execution_wall_per_sim_s'];readers=old['native_reader_wall_per_sim_s']
    run_hours=sum(remaining[a]*480*(rates[a]+readers[a]) for a in remaining)/3600
    origin=json.loads((base/'wall_budget_origin.json').read_text())['started_unix_s']
    elapsed=(time.time()-origin)/3600
    video_hours=old['scenarios']['1']['remaining_video_h']
    # Keep prior 15% contingency; add one wall hour for replay regeneration and
    # hash capture overhead pending the initialization decision, not a new run.
    future=(run_hours+video_hours)*1.15+1.
    logs={};rgb_rates={}
    for arm,row in old['pilots'].items():
        keep=images=0
        for parent,dirs,files in os.walk(row['root'],followlinks=False):
            dirs[:]=[d for d in dirs if d!='sealed' and not d.startswith('sealed_')]
            for name in files:
                if name=='sealed_test.json':continue
                p=Path(parent)/name
                if p.is_symlink():continue
                size=p.stat().st_blocks*512
                if name.endswith('.png'):images+=size
                elif name.endswith('.pkl'):continue
                else:keep+=size
        duration=row['result']['simulated_s']
        logs[arm]=keep/duration
        rgb_rates[arm]=images/duration
    # Include 8 KiB of metadata per 10-Hz paired acquisition beyond existing logs.
    hash_metadata_rate=8192*10
    storage={a:remaining[a]*480*(logs[a]+hash_metadata_rate) for a in remaining}
    # C0 has no source images: retain its RGB and native depth losslessly.
    # Native depth plus recorded noise recipe reconstructs the consumed packet;
    # both native and noisy packet hashes are retained.
    maximum_depth_rate=measurement['maximum_bytes_per_camera_frame']*2*10
    storage['C0']+=remaining['C0']*480*(max(rgb_rates.values())+maximum_depth_rate)
    videos=4*1024**3
    required=sum(storage.values())*1.15+videos
    usable=shutil.disk_usage(base).free-protocol['caps']['recovery_reserve_bytes']-128*1024**2
    report=dict(schema='navigation_capability_resume_budget.v1',date='2026-09-26',
        elapsed_conservative_wall_hours=elapsed,remaining_science_episodes=remaining,
        pre_fix_pilots_not_counted_in_corrected_harness_screen_or_gate=True,
        validation_episodes_per_maze=1,validation_ids=old['fixed_validation_episode_ids'],
        future_runs_and_native_analysis_hours=run_hours,video_hours=video_hours,
        contingency_fraction=.15,extra_replay_and_capture_allowance_hours=1.,
        projected_total_wall_hours=elapsed+future,wall_cap_hours=120,
        time_fits=elapsed+future<=120,
        retention={a:results[a]['retention'] for a in results},
        non_image_log_allocated_bytes_per_sim_s=logs,extra_hash_metadata_bytes_per_sim_s=hash_metadata_rate,
        full_retention_rgb_bytes_per_sim_s=max(rgb_rates.values()),
        fallback_depth_measurement=measurement,full_retention_depth_bytes_per_sim_s=maximum_depth_rate,
        remaining_storage_bytes_by_controller=storage,video_storage_allowance_bytes=videos,
        additional_storage_with_contingency_bytes=required,recovery_usable_bytes=usable,
        storage_fits=required<=usable,
        assumptions=['All future missions use the full 480-s budget.',
            'One corrected C1 screen and one complete C0 gate; later harness versions require re-projection.',
            'Only previously verified C3 concurrency contributes a speedup.',
            'C4 maximum observed depth archive size sizes C0 fallback; it is an estimate, not an upper bound on all unseen views.',
            'Future recording writes remain subject to live reserve admission.',
            'The pending initialization choice may add compute; re-project before launch if it does.'],
        cohort_launch_authorized_by_this_report=False,
        blocker=('Full C0 RGB-D fallback exceeds usable storage. ' if required>usable else '')+
            'Shared visual-start anchoring and its post-fix test remain unfinished; no new cohort launched.')
    destination=repo/'docs/go2_navigation_capability_resume_budget_2026-09-26.json'
    with destination.open('x') as f:json.dump(report,f,indent=2)
    print(output.dumps({k:report[k] for k in ['projected_total_wall_hours','time_fits','additional_storage_with_contingency_bytes','recovery_usable_bytes','storage_fits','blocker']}))


if __name__=='__main__':main()
