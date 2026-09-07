"""Raw causal replay, unchanged strict visibility, separate footprint accounting."""
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.independent_layout_collection_audit_development import audit_condition
from scripts.ordered_dynamic_pilot_development import whole_stream_witness
from scripts.startup_raw_sensor_audit_development import read_json,read_npz


def audit_dynamic_condition(directory,spec,result,definition):
    report,prefix,_,_=audit_condition(directory,spec,result,definition)
    cameras=read_json(directory,'camera_audit.json');footprints=[];rasters=[]
    for i,c in enumerate(cameras):
        row=read_json(directory,f'raster_{i:04d}.json')
        assert row['physical_sample_index']==c['physical_sample_index']
        assert row['order']['order']=='floor_first' and row['order']['roles']==['floor','walls']
        assert set(row['order']['surfaces'])=={'floor','walls'}
        precision=row['precision'];positions=precision['rgb_target_sample_positions']
        assert 1<=precision['subpixel_bits']<=32 and 1<=precision['depth_target_depth_bits']<=64
        assert 1<=precision['rgb_target_samples']<=32 and len(positions)==precision['rgb_target_samples']
        assert all(len(p)==2 and all(0<=x<=1 for x in p) for p in positions)
        if rasters:assert row['order']==rasters[0]['order'] and precision==rasters[0]['precision']
        rasters.append(row)
        native=read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m']
        score=evaluate_footprint(native,spec['geometry']['wall_boxes'],c['world_from_optical'],render_near_m=.005)
        assert score['original_strict_score']==report['depth_checks'][i]['physical_visibility']
        footprints.append(dict(frame=i,score=score))
    return dict(report=report,prefix=prefix,whole_stream=whole_stream_witness(directory,result),
        raster_readbacks=rasters,footprint_diagnostics=footprints,
        training_eligibility_granted=False,navigation_qualified=False)
