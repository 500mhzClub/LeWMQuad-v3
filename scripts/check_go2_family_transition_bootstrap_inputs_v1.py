"""New motion/contact prediction scope; whole-population policy stream validation."""
from collections import Counter
import json
import resource
import time
import cv2
import torch

from lewm.family_transition_bootstrap_scope_development import admit_scope
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.cumulative_pulse_contact_development import CumulativePulseRGBBodyJEPA
from lewm.cumulative_pulse_learning_development import training_loss
from lewm.pulse_timed_rgb_body_jepa_development import validate_timed_plan
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.geometry_progress_family_policy_stream_development import FamilyPolicyStream
from scripts.read_go2_geometry_progress_family_causal_v1 import OUTPUT as DERIVATION, INPUT
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_family_transition_bootstrap_inputs_v1_attempt_001'
PROTOCOL = 'docs/go2_family_transition_bootstrap_inputs_v1_2026-09-08.md'
COLLECTION_SHA = '376b2eeacfd5e741ba6b7a0e1b5e04399f782d16943d66ffdff92eada93ef0fb'
CAUSAL_SHA = '37bd88d43fd0ebdcee80282d38695f82a41b83313d08158d076f5ad4ce7145a7'
SCHEDULE_SEED = 2026091001


def main():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive new transition input check; no retry/resume')
    causal_ids = {'result.json': CAUSAL_SHA}; collection_ids = {'result.json': COLLECTION_SHA}
    verify_artifacts(DERIVATION, causal_ids); verify_artifacts(INPUT, collection_ids)
    derived, collection = read_json(DERIVATION, 'result.json'), read_json(INPUT, 'result.json')
    scope = admit_scope(collection, derived)
    if derived['collection_sha256'] != COLLECTION_SHA or derived['collection_root'] != str(INPUT):
        raise ValueError('exact new-scope collection/derivation binding required')
    causal_ids |= derived['artifact_sha256']; collection_ids |= collection['artifact_sha256']
    verify_artifacts(DERIVATION, causal_ids); verify_artifacts(INPUT, collection_ids)
    original = read_json(INPUT, 'launch.json'); verify(original)
    sources = discover_sources((PROTOCOL, 'scripts/check_go2_family_transition_bootstrap_inputs_v1.py',
        'lewm/tests/test_family_transition_bootstrap_scope_development.py',
        'lewm/tests/test_geometry_progress_family_learning_view_development.py',
        'lewm/tests/test_geometry_progress_family_policy_stream_development.py'), derived['source_sha256'])
    definition = original | dict(source_sha256=sources)
    verify(definition)
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+256*1024**2:
        raise ValueError('input-check RAM/storage allowance unavailable')
    view = FamilyWindowView(read_json(DERIVATION, 'windows.json'))
    stream = FamilyPolicyStream(view, output=INPUT, bindings=collection['artifact_sha256'],
        tensor_index=read_json(DERIVATION, 'tensor_index.json'))
    schedule = view.schedule(updates=1200, batch_size=6, seed=SCHEDULE_SEED)
    draws = Counter(view.windows[i]['trial'] for b in schedule['batches'] for i in b)
    if len(draws) != 48 or set(draws.values()) != {150}:
        raise ValueError('equal complete training-episode draws required')
    if len(view.indices('train')) != 336 or len(view.indices('geometry_transfer')) != 348:
        raise ValueError('complete actual role populations required')
    torch.set_num_threads(1); cv2.setNumThreads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026091002); model = CumulativePulseRGBBodyJEPA(32).eval()
    before = state_digest(model.state_dict())
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, collection_sha256=collection_ids,
        causal_sha256=causal_ids, protocol=PROTOCOL, scope=scope, hardware=resources,
        untrained_model_sha256=before, optimizer_steps=0, threads=1))
    write_json(OUTPUT/'training_schedule.json', schedule)
    print('TRANSITION_INPUTS_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter(); stats = {}
    try:
        with (OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
            for role in ('train', 'geometry_transfer'):
                indices = view.indices(role); active_count = motion_count = contact_count = positives = future_count = 0
                for at in range(0, len(indices), 6):
                    batch_ids = indices[at:at+6]
                    if role == 'train':
                        batch = stream.training_batch(batch_ids); inputs = batch['inputs']
                    else:
                        inputs = stream.inference_batch(batch_ids, role=role)
                    active, offsets = validate_timed_plan(inputs['known_action_blocks'], inputs['known_action_valid'], len(batch_ids))
                    with torch.inference_mode():
                        out = model(**inputs)
                        if role == 'train':
                            loss, _ = training_loss(model, batch, 'jepa')
                            if not torch.isfinite(loss): raise ValueError('finite full target contract required')
                    if not torch.equal(out['prediction_valid'], active) or not torch.equal(out['target_offsets_ns'], offsets):
                        raise ValueError('wrong returned causal horizons')
                    if not all(torch.isfinite(out[k][active]).all() for k in ('direct_outcomes', 'rollout_outcomes')):
                        raise ValueError('nonfinite prediction interface')
                    active_count += int(active.sum())
                    if role == 'train':
                        t = batch['targets']; motion_count += int(t['motion_valid'].sum())
                        contact_count += int(t['contact_valid'].sum()); future_count += int(t['future_valid'].sum())
                        positives += int(t['contact'][t['contact_valid']].sum())
                    if at % 60 == 0:
                        monitor.write(json.dumps(dict(role=role, windows_completed=at+len(batch_ids), **hardware()))+'\n'); monitor.flush()
                stats[role] = dict(materialized_windows=len(indices), active_prediction_slots=active_count,
                    training_target_motion_slots=motion_count, training_target_contact_slots=contact_count,
                    training_target_positive_contact_slots=positives, training_target_future_images=future_count,
                    future_images_read_for_inputs=False, future_images_read_as_training_targets=role=='train')
                print('TRANSITION_INPUT_ROLE_COMPLETE', role, stats[role], flush=True)
        if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('input validation changed model or gradients')
        verify(definition); verify_artifacts(INPUT, collection_ids); verify_artifacts(DERIVATION, causal_ids)
        write_json(OUTPUT/'result.json', dict(status='FAMILY_TRANSITION_BOOTSTRAP_INPUTS_COMPLETE',
            scope=scope, roles=stats, original_readiness_value=False, original_task_design_value=False,
            original_failures_preserved=True,
            original_outcomes_changed=False, collection_result_sha256=COLLECTION_SHA, causal_result_sha256=CAUSAL_SHA,
            training_schedule_sha256=schedule['schedule_sha256'], training_episode_draw_counts=dict(draws),
            optimizer_steps=0, untrained_model_sha256=before, model_trained=False,
            wall_s=time.perf_counter()-start, maximum_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            source_sha256=sources, artifact_sha256={n: digest(OUTPUT/n) for n in ('launch.json', 'training_schedule.json', 'resource_monitor.jsonl')},
            transition_fit_inputs_validated=True, native_execution=False, navigation_qualified=False, goal_achieved=False))
        print('TRANSITION_INPUTS_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_TRANSITION_INPUT_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
