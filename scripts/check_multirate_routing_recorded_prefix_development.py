"""Recorded routing updates at 10 Hz and 2.5 Hz; no command dispatch."""
from contextlib import closing
import json
from pathlib import Path
import time

import cv2
import numpy as np

from lewm.multirate_routing_map_development import MultirateRoutingMap
from scripts import profile_stop_conditioned_early_decisions_development as source

OUTPUT = Path('docs/go2_multirate_routing_prefix_v2_2026-09-13.json')
FAILURE = OUTPUT.with_suffix('.failure.json')


def decode_evidence(value):
    if isinstance(value, dict):
        return {k:tuple(v) if k == 'identity' and isinstance(v, list) else decode_evidence(v)
            for k,v in value.items()}
    if isinstance(value, list): return [decode_evidence(v) for v in value]
    return value


def main():
    assert not OUTPUT.exists() and not FAILURE.exists()
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    maps = {'full_rate':MultirateRoutingMap(), 'quarter_rate':MultirateRoutingMap()}
    reader = source.packets.ExtendedReturnBudgetRGBDReplay(source.INPUT)
    auxiliary = json.loads((source.INPUT/'auxiliary_camera_audit.json').read_text())
    rows = []; retained = None
    try:
        with closing(source.packets.read_rows(source.INPUT)) as recorded:
            for frame in range(61):
                original = next(recorded)['decision']; p,d,_,now = reader.packet(frame)
                _,aux = source.packets.rgb_packet(source.INPUT,frame,p,
                    source.public_acquisition(auxiliary[frame]),now_ns=now)
                record = dict(frame=frame, measured_ns=now)
                for label, mapper in maps.items():
                    if label == 'quarter_rate' and frame%4: continue
                    started = time.perf_counter()
                    snap = mapper.update(p,d,decode_evidence(original['evidence']),auxiliary_depth=aux,measured_ns=now)
                    record[label+'_s'] = time.perf_counter()-started
                    assert snap.frame == frame and snap.measured_ns == now
                    if label == 'full_rate':
                        expected = original['memory_receipt']
                        assert snap.primary_current_floor_cells == expected['current_observed_floor_cells']
                        assert snap.auxiliary_current_floor_cells == expected['auxiliary_receipt']['current_observed_floor_cells']
                        assert len(snap.floor) == expected['retained_observed_floor_cells']
                        assert len(snap.occupied) == expected['retained_occupied_cells']
                        assert snap.floor_height == expected['floor_height_map_m']
                        np.testing.assert_array_equal(snap.map_from_initial,expected['map_from_initial'])
                        record['recorded_map_counts_and_reference_equal'] = True
                full, quarter = maps['full_rate'].latest, maps['quarter_rate'].latest
                assert quarter.floor <= full.floor and quarter.occupied <= full.occupied
                assert quarter.age_ns(now_ns=now) == frame%4*100_000_000
                if retained is None: retained = quarter
                assert retained.frame == 0 and retained.measured_ns == 1_500_000_000
                record.update(sparse_map_frame=quarter.frame,sparse_age_ns=quarter.age_ns(now_ns=now),
                    full_floor_cells=len(full.floor),sparse_floor_cells=len(quarter.floor),
                    full_occupied_cells=len(full.occupied),sparse_occupied_cells=len(quarter.occupied))
                selection = original.get('new_selection')
                if selection and 'proposal' in selection:
                    goal = original['mission_receipt']['active_goal_initial_body_xy_m']
                    route = full.route(goal); sparse_route = quarter.route(goal)
                    expected = selection['proposal']
                    record['recorded_route_cells_equal'] = route['route_cells'] == expected['route_cells']
                    record['recorded_route_status_equal'] = route['status'] == expected['status']
                    record['sparse_route_cells_equal_full'] = sparse_route['route_cells'] == route['route_cells']
                    assert route['command_authorized'] is False and sparse_route['command_authorized'] is False
                rows.append(record)
                if frame%20 == 0: print('ROUTING_MAP_FRAME',frame,flush=True)
        report = dict(status='MULTIRATE_ROUTING_PREFIX_COMPLETE', observations=61, rows=rows,
            update_counts={label:sum(label+'_s' in r for r in rows) for label in maps},
            mean_update_ms={label:float(np.mean([r[label+'_s'] for r in rows if label+'_s' in r]))*1000 for label in maps},
            max_sparse_age_ms=max(r['sparse_age_ns'] for r in rows)/1e6,
            actual_acquisition_timestamps_retained=True, recorded_public_pose_only=True,
            full_rate_recorded_counts_equal=True, full_cell_identity_against_recording_proven=False,
            sparse_map_subset_of_full_rate=True, old_snapshot_unchanged=True,
            routing_only=True, footprint_or_contact_coverage_provided=False,
            native_pose_loaded=False, native_execution=False, model_loaded=False,
            sensor_and_pose_processing_included_in_timing=False, continuous_control_qualified=False,
            input=str(source.INPUT), sources={p:source.digest(Path(p)) for p in (
                'lewm/multirate_routing_map_development.py',
                'scripts/check_multirate_routing_recorded_prefix_development.py')})
        with OUTPUT.open('x') as out: json.dump(report,out,indent=2);out.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    except BaseException as error:
        with FAILURE.open('x') as out: json.dump(dict(reason=repr(error),completed_rows=rows),out,indent=2)
        raise


if __name__ == '__main__': main()
