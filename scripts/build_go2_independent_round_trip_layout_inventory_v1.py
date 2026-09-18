"""Persist source-only new maze construction with all structural exclusions."""
import sys
import numpy as np
from lewm.independent_round_trip_layouts_development import build_inventory,validate_inventory,pack,public_mission
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json,verify
from scripts.startup_source_inventory_development import discover_sources

OUTPUT=BASE/'go2_independent_round_trip_layout_inventory_v1_attempt_001'
SOURCE='scripts/build_go2_independent_round_trip_layout_inventory_v1.py'
PROTOCOL='docs/go2_independent_round_trip_layout_inventory_v1_2026-09-10.md'
TEST='lewm/tests/test_independent_round_trip_layouts_development.py'


def main():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive source inventory required')
    sources=discover_sources((SOURCE,PROTOCOL,TEST,'AGENTS.md','.ignore','config/go2_platform_manifest.yaml'),{});verify(sources)
    inventory=build_inventory();validate_inventory(inventory)
    for spec in inventory['layouts']:
        definition=pack(spec);mission=public_mission(spec['layout_index'])
        if len(definition.static_objects)!=len(spec['geometry']['wall_boxes']):raise ValueError('complete scene wall roster required')
        if set(mission)!={'goal_initial_body_xy_m','return_initial_body_xy_m','require_return_after_goal'}:
            raise ValueError('coordinate-only public mission required')
    verify(sources);create_output(OUTPUT)
    launch=dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),native_execution=False,
        runtime_artifacts_read=False,model_loaded=False,source_defined_construction_only=True,
        python_version=sys.version,numpy_version=np.__version__)
    write_json(OUTPUT/'launch.json',launch)
    try:
        write_json(OUTPUT/'inventory.json',inventory);verify(sources)
        ids={n:digest(OUTPUT/n) for n in ('launch.json','inventory.json')};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='INDEPENDENT_ROUND_TRIP_LAYOUT_INVENTORY_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,layouts=8,prior_source_layouts=48,
            prior_abstract_groups=inventory['prior_abstract_topology_groups'],candidates_examined=inventory['candidates_examined'],
            structural_rejections=len(inventory['structural_rejections']),
            exact_abstract_topology_and_grid_disjointness=True,native_execution=False,model_loaded=False,
            physical_geometry_verified=False,final_evaluation=False,navigation_qualified=False,goal_achieved=False))
        print('INDEPENDENT_ROUND_TRIP_INVENTORY_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_INDEPENDENT_ROUND_TRIP_INVENTORY_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
