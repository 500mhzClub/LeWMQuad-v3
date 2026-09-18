"""Fixed within-layout startup pairing, with explicit incomplete early stops."""
import re
import math

from lewm.independent_round_trip_comparison_study_development import CASES, require_case
from lewm.independent_round_trip_multiarm_contract_development import require_collection
from scripts.all_phase_residual_maze02_startup_development import compare_startup
from scripts.navigation_artifact_root_development import validate_root

MATCHED = 'EXACT_PRECOMMAND_STARTUP'
UNAVAILABLE = 'UNAVAILABLE_EARLY_STOP'


def reference_case(case):
    require_case(case)
    return CASES[4*case.layout_index]


def definition(case, reference, current):
    first = reference_case(case)
    require_collection(first, reference); require_collection(case, current)
    counts = []; unavailable = []
    for selected, collection in ((first, reference), (case, current)):
        row = dict(case=selected.name)
        for key in ('physics_samples', 'decisions', 'completed_ticks'):
            value = collection[key]
            if type(value) is not int or value < 0:
                raise ValueError('nonnegative actual startup population counts required')
            row[key] = value
        if row['physics_samples'] < 900 or row['decisions'] < 4 or row['completed_ticks'] < 3:
            if collection['physical_stop'] is None and collection['acquisition_stop'] is None:
                raise ValueError('short startup requires a recorded physical or acquisition stop')
            unavailable.append(selected.name)
        counts.append(row)
    return dict(layout_index=case.layout_index, reference_case=first.name, candidate_case=case.name,
        collection_counts=counts, unavailable_cases=sorted(set(unavailable)),
        status=UNAVAILABLE if unavailable else MATCHED,
        physical_and_public_startup_exact=False if unavailable else True,
        later_physical_outcomes_compared=False, unexecuted_outcomes_inferred=False)


def compare_case_startup(output, case, reference, current):
    receipt = definition(case, reference, current)
    if receipt['status'] == UNAVAILABLE:
        return receipt
    output = validate_root(output)
    details = compare_startup(output/reference_case(case).name, output/case.name)
    for owner, replacement in (('original', 'reference'), ('candidate', 'candidate')):
        for field in ('command', 'terminal'):
            details[replacement+'_first_command' + ('_terminal' if field == 'terminal' else '')] = details.pop(
                owner+'_first_model_'+field)
    receipt.update(details)
    require_startup(case, reference, current, receipt)
    return receipt


def require_startup(case, reference, current, receipt):
    expected = definition(case, reference, current)
    for key, value in expected.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise ValueError('fixed paired startup receipt differs: '+key)
    if expected['status'] == UNAVAILABLE:
        if receipt != expected:
            raise ValueError('incomplete startup cannot claim additional matched evidence')
        return
    required = dict(common_prefix_frames=4, physical_prefix_samples=900,
        completed_zero_warmup_commands=3, model_forecasts_required_equal=False,
        post_warmup_controller_state_required_equal=False)
    for key, value in required.items():
        if type(receipt.get(key)) is not type(value) or receipt[key] != value:
            raise ValueError('complete paired pre-command startup required: '+key)
    for key in ('raw_physics_prefix_sha256', 'public_startup_sha256'):
        if not isinstance(receipt.get(key), str) or re.fullmatch('[0-9a-f]{64}', receipt[key]) is None:
            raise ValueError('exact raw startup fingerprints required')
    for owner in ('reference', 'candidate'):
        if owner+'_first_command' not in receipt or owner+'_first_command_terminal' not in receipt:
            raise ValueError('first command/terminal evidence required without imposing equality')
        command = receipt[owner+'_first_command']; terminal = receipt[owner+'_first_command_terminal']
        if (type(command) is not list or len(command) != 3 or
                any(type(value) not in (int, float) or not math.isfinite(value) for value in command)
                or (terminal is not None and type(terminal) is not str)):
            raise ValueError('finite first command and explicit terminal required')
    if reference_case(case) == case:
        if (receipt['reference_first_command'] != receipt['candidate_first_command'] or
                receipt['reference_first_command_terminal'] != receipt['candidate_first_command_terminal']):
            raise ValueError('self-reference must describe the same first command and terminal')
