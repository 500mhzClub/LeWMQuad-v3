"""Freeze explicit prospective layout/context roles; no physics/training/runtime."""
import shutil
from lewm.independent_layout_inventory_development import build_inventory,validate_inventory
from lewm.independent_layout_inventory_audit_development import audit_inventory
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.navigation_artifact_root_development import BASE,verify_artifacts

PREVIOUS=BASE/'go2_independent_pulse_context_pilot_v1_attempt_001'
OUTPUT=ROOT/'.generated/go2_independent_layout_inventory_v1_attempt_001'
PROTOCOL='docs/go2_independent_layout_inventory_v1_2026-09-06.md'
IDENTITIES={
    'launch.json':'bb06bb68d6cc8702c224d1a5b78d5b4285b2c3dcf6636e9c029f425d0519319d',
    'result.json':'132bcf46e4c62d892765b8c61a1a38563aa2aeeacc5fa2e9d1896f27e8f567ae',
    'context_encoding_correction_v1_launch.json':'56aee0c0345cba37473d6c79bc7812aac0a7b34254ad25be4ecbc1adfcc492b1',
    'context_encoding_correction_v1.json':'4b51738c6ae50051373b407685181b6ef18598ae528701a70b11270069416399',
    'context_encoding_correction_v1_targets.json':'7aed9a7f1d9ad9602309887c0ba6a859700f702fc46868181f816dd31ea91dcf',
    'context_audit_failure.json':'95883bdb2bed71d157c100b3b7566a6030420ab31a977462b5fbefba937ccdbe',
}


def preflight():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive inventory constructor; no overwrite')
    if shutil.disk_usage(ROOT).free<10*1024**3+100*1024**2:raise ValueError('inventory storage plus10GiB reserve required')
    verify_artifacts(PREVIOUS,IDENTITIES)
    inherited=read_json(PREVIOUS,'context_encoding_correction_v1_launch.json')['source_sha256'];verify_bindings(inherited)
    sources=discover_sources((PROTOCOL,'scripts/build_go2_independent_layout_inventory_v1.py',
        'lewm/tests/test_independent_layout_inventory_development.py'),inherited)
    verify_bindings(sources)
    return dict(source_sha256=sources,predecessor_root=str(PREVIOUS),predecessor_sha256=IDENTITIES,
        output_root=str(OUTPUT),physics_collected=False,training_launched=False,final_evaluation=False,
        planned_episode_count=1440,planned_layout_count=12,minimum_free_bytes=10*1024**3)


def main():
    launch=preflight();OUTPUT.mkdir();write_json(OUTPUT/'launch.json',launch)
    try:
        inventory=build_inventory();validate_inventory(inventory);audit=audit_inventory(inventory)
        write_json(OUTPUT/'inventory.json',inventory);write_json(OUTPUT/'structural_audit.json',audit)
        # Read back the serialized manifest; equality includes scalar types and all episode definitions.
        validate_inventory(read_json(OUTPUT,'inventory.json'))
        verify_bindings(launch['source_sha256']);verify_artifacts(PREVIOUS,IDENTITIES)
        write_json(OUTPUT/'result.json',dict(status='INDEPENDENT_LAYOUT_INVENTORY_FROZEN',
            output_sha256={p:digest(OUTPUT/p) for p in ('inventory.json','structural_audit.json')},
            layouts=12,planned_episodes=1440,role_counts=inventory['role_counts'],episode_role_counts=inventory['episode_role_counts'],
            graph_pairs_independently_checked=audit['layout_pairs_checked'],physics_collected=False,model_trained=False,
            native_setup_verified=False,positive_contact_coverage_verified=False,final_evaluation=False,goal_achieved=False))
        print('INVENTORY_FROZEN',inventory['role_counts'],inventory['episode_role_counts'],flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_INVENTORY_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
