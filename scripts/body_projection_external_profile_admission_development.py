"""Reconstruct the completed body-projection witness without writing it again.

The original checker executes unchanged with private output and CLI bindings.
This module does not launch a replay, create an output directory or select data.
"""
import hashlib
import json
from types import FunctionType, SimpleNamespace

from scripts import verify_go2_body_projected_tiled_controller_completion_v1 as completed
from scripts.external_body_projection_profile_replay_development import OUTPUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_raw_sensor_audit_development import read_json

VERIFICATION = 'docs/go2_body_projected_tiled_controller_completion_verification_2026-09-11.json'
VERIFICATION_SHA = '47c6e00d30f9b7e8c31b19dbeed511699d7565145e2b34273fe56e3c96eb25ab'
RESULT_SHA = '548c8afe30d87c5f2d503c820a77a03c75d7c764dd6d045529d1a258fad94eea'
EXECUTION_SHA = '39755d9dbec68b6fd5525dd9389c741a22d4e7ae0e30626a1fbe7fe91092e351'
LAUNCH_SHA = '69dd113461f8301d6d47daf9f0174441aed8f5047a27dd68de1c5103f18c652f'


def reference_witness():
    verify({VERIFICATION: VERIFICATION_SHA, completed.EXECUTION: EXECUTION_SHA})
    witness = json.loads((ROOT/VERIFICATION).read_text())
    if (witness['status'] != 'BODY_PROJECTED_TILED_COMPLETION_VERIFIED'
            or witness['result_sha256'] != RESULT_SHA
            or witness['execution_sha256'] != EXECUTION_SHA
            or witness['artifact_sha256']['launch.json'] != LAUNCH_SHA):
        raise ValueError('exact completed body-projection identities required')
    verify(witness['source_sha256'])
    return witness


def capture_verification():
    destination = OUTPUT/'read_only_completion_capture.json'
    receipts = []

    def capture(path, value):
        if path != destination or receipts:
            raise ValueError('exactly one original completion receipt required')
        receipts.append(value)

    def capture_digest(path):
        if path == destination:
            if len(receipts) != 1:
                raise ValueError('completion receipt must precede logging digest')
            payload = json.dumps(receipts[0], indent=2, allow_nan=False)+'\n'
            return hashlib.sha256(payload.encode()).hexdigest()
        return completed.digest(path)

    class FixedArguments:
        def __init__(self):
            self.arguments = []

        def add_argument(self, *args, **kwargs):
            expected = ('--result-sha256', '--execution-sha256')
            if (len(self.arguments) >= len(expected)
                    or args != (expected[len(self.arguments)],)
                    or kwargs != {'required': True}):
                raise ValueError('unchanged two-argument frozen checker CLI required')
            self.arguments.append(args[0])

        def parse_args(self):
            if self.arguments != ['--result-sha256', '--execution-sha256']:
                raise ValueError('both original actual result and execution arguments required')
            return SimpleNamespace(result_sha256=RESULT_SHA, execution_sha256=EXECUTION_SHA)

    function = completed.main
    if function.__closure__ is not None:
        raise ValueError('closure-free original checker required')
    bindings = dict(OUTPUT=destination, write_json=capture, digest=capture_digest,
        argparse=SimpleNamespace(ArgumentParser=FixedArguments), print=lambda *a, **k:None)
    clone = FunctionType(function.__code__, function.__globals__ | bindings,
        function.__name__, function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone()
    if len(receipts) != 1:
        raise ValueError('exactly one original completion receipt required')
    return receipts[0]


def admit_completed():
    witness = reference_witness()
    actual = capture_verification()
    if {k:v for k,v in actual.items() if k != 'utc'} != {k:v for k,v in witness.items() if k != 'utc'}:
        raise ValueError('complete original witness including failures must reconstruct')
    result = read_json(completed.run.OUTPUT, 'result.json')
    launch = read_json(completed.run.OUTPUT, 'launch.json')
    rows = [json.loads(line) for line in (completed.run.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    # Authenticate returned data again after reading, rather than relying on the
    # earlier checker admission across a read/modify race.
    completed.verify_artifacts(completed.run.OUTPUT,
        witness['artifact_sha256'] | {'result.json': RESULT_SHA, 'launch.json': LAUNCH_SHA})
    if reference_witness() != witness:
        raise ValueError('original completion changed during admission')
    return witness, result, launch, rows
