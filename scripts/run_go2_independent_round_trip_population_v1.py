"""Final entry point for an explicitly reviewed original independent study.

No independent execution is possible without the completed five-stage input
bundle and exact final policy review. Source preflight consumes neither.
"""
import argparse
from copy import deepcopy
import json
import re

from scripts import independent_round_trip_final_admission_development as admission
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json

SOURCE='scripts/run_go2_independent_round_trip_population_v1.py'
TEST='lewm/tests/test_independent_round_trip_final_launcher_development.py'
PROTOCOL='docs/go2_independent_round_trip_population_execution_v1_2026-09-11.md'
BUNDLE='docs/go2_independent_round_trip_completed_input_bundle_v1.json'
OUTPUT=BASE/'go2_independent_round_trip_population_v1_attempt_001'
TEMPLATE_KEYS=('input_sha256','native_sha256','native_scene_sha256','native_geometry_sha256',
    'opencv_binary_sha256','opencv_version','rules','renderer_environment')


def prepared_sources():
    sources=discover_sources((SOURCE,TEST,PROTOCOL),admission.prepared_sources())
    verify(sources)
    return sources


def prepare_input_bundle(batch_sha,waiter_sha,*,sources):
    """Complete expensive input checks once for a subsequently written review."""
    admission.queue.owners_ended()
    if (set(waiter_sha)!= {'frontier','hold','contact','tracking','budget'}
            or any(type(v) is not str or re.fullmatch('[0-9a-f]{64}',v) is None
                   for v in [batch_sha,*waiter_sha.values()])):
        raise ValueError('exact completed batch and five waiter identities required')
    verify(sources)
    inputs=admission.inputs.admit(batch_sha,sources)
    first=admission.queue.original.original.admit({k:waiter_sha[k] for k in ('frontier','hold','contact')},
        adapter_batch_result_sha256=batch_sha,sources=sources,full=True)
    second=admission.queue.original.admit(first,waiter_sha['tracking'],sources=sources,full=True)
    queue=admission.queue.admit(second,waiter_sha['budget'],sources=sources,full=True)
    admission.require_links(inputs,queue)
    return dict(schema='independent_round_trip_completed_input_bundle.v1',
        source_sha256=deepcopy(sources),input_admission=inputs,five_stage_queue_admission=queue,
        final_policy_review_completed=False,population_execution_permitted=False)


def make_launch(bundle,sources,review_sha,bundle_sha):
    if (set(bundle)!= {'schema','source_sha256','input_admission','five_stage_queue_admission',
                     'final_policy_review_completed','population_execution_permitted'}
            or bundle['schema']!='independent_round_trip_completed_input_bundle.v1'
            or bundle['final_policy_review_completed'] is not False
            or bundle['population_execution_permitted'] is not False
            or any(sources.get(n)!=h for n,h in bundle['source_sha256'].items())
            or sources.get(BUNDLE)!=bundle_sha or sources.get(admission.REVIEW)!=review_sha):
        raise ValueError('exact source-bound completed input bundle and separate review required')
    inputs=bundle['input_admission'];queue=bundle['five_stage_queue_admission']
    admission.require_links(inputs,queue)
    verify_artifacts(admission.inputs.batch.OUTPUT,{'launch.json':admission.inputs.BATCH_LAUNCH})
    template=read_json(admission.inputs.batch.OUTPUT,'launch.json')
    overlap=admission.overlap
    launch={k:deepcopy(template[k]) for k in TEMPLATE_KEYS}
    launch.update(source_sha256=deepcopy(sources),protocol=PROTOCOL,output_root=str(OUTPUT),
        input_admission=deepcopy(inputs),five_stage_queue_admission=deepcopy(queue),
        completed_input_bundle=dict(path=BUNDLE,sha256=bundle_sha),
        policy_review=dict(path=admission.REVIEW,sha256=review_sha),
        runtime_verifier=dict(source=admission.SOURCE,function='verify_population'),
        overlap_verifier=dict(source=overlap.SOURCE,function='verify_overlap'),
        population_entrypoint=deepcopy(overlap.ENTRYPOINT),
        ordered_cases=admission.inputs.study.manifest()['ordered_cases'],
        runtime=deepcopy(admission.runtime.FIXED_RUNTIME),staged_runtime=deepcopy(overlap.driver.staged.FIXED),
        audit_cpu_monitor=deepcopy(overlap.monitored.FIXED),overlap_evidence=deepcopy(overlap.EVIDENCE),
        robot_urdf_sha256=digest(admission.runtime.URDF),final_policy_review_completed=True,
        complete_input_admission_performed=True,native_queue_completion_verified=True,
        native_execution=True,model_training=False,navigation_qualified=False,
        real_time_qualified=False,hardware_qualified=False,goal_achieved=False)
    return launch


def resources():
    measured=admission.runtime.hardware()
    result=admission.inputs.study.resources_for(measured)
    if measured['memory_available_bytes']<admission.overlap.driver.staged.FIXED['available_memory_minimum_bytes']:
        raise ValueError('64GiB available RAM required for the bounded staged driver')
    return dict(hardware=measured,resource_admission=result)


def main():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--source-preflight-only',action='store_true')
    modes.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--bundle-sha256');parser.add_argument('--review-sha256')
    args=parser.parse_args()
    admission.overlap.monitored.require_environment()
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive whole study; no retry or resume')
    sources=prepared_sources();measured=resources()
    if args.source_preflight_only:
        print('INDEPENDENT_POPULATION_SOURCE_PREFLIGHT_PASS',len(sources),json.dumps(measured),flush=True);return
    admission.queue.owners_ended()
    if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}',v) is None
           for v in (args.bundle_sha256,args.review_sha256)):
        raise ValueError('exact completed bundle and final review SHA-256 required')
    documents={BUNDLE:args.bundle_sha256,admission.REVIEW:args.review_sha256}
    verify(documents)
    sources=discover_sources(tuple(documents),sources);verify(sources)
    bundle=json.loads((ROOT/BUNDLE).read_text())
    launch=make_launch(bundle,sources,args.review_sha256,args.bundle_sha256)
    admission.verify_population(launch,full=True)
    admission.overlap.verify_overlap(launch,full=True)
    measured=resources();launch.update(measured)
    verify(sources);verify(documents);admission.queue.owners_ended()
    if args.preflight_only:
        print('INDEPENDENT_POPULATION_COMPLETE_PREFLIGHT_PASS',len(sources),flush=True);return
    admission.runtime.require_native_idle()
    create_output(OUTPUT)
    try:
        write_json(OUTPUT/'launch.json',launch);sha=digest(OUTPUT/'launch.json')
        print('INDEPENDENT_POPULATION_LAUNCHED',sha,flush=True)
        result=admission.overlap.run_population(OUTPUT,sha,admission.verify_population)
        print('INDEPENDENT_POPULATION_RETURNED',result['status'],digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        path=OUTPUT/'failure.json'
        if not path.exists() and not path.is_symlink():
            write_json(path,dict(status='TERMINAL_INDEPENDENT_POPULATION_LAUNCH_FAILURE',
                reason=repr(error),automatic_retry=False,evidence_preserved=True))
        raise


if __name__=='__main__':main()
