"""Physical outcomes and actual model-input receipts for the RGB pilot."""
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_multiseed_navigation_development as previous
from scripts import run_go2_neural_rgb_transfer_development as experiment


def evaluate(index, arm):
    root = previous.previous.path(experiment.ROOT.format(index=index, arm=arm))
    variant = 'no_rgb' if '_no_rgb_' in arm else 'full'
    plans = [p for p in previous.previous.read(root, 'planning.json') if 'selection' in p]
    counts = []
    for plan in plans:
        receipt = plan['selection']['model_input_treatment']
        count = receipt['rgb_nonzero_values']
        if (receipt['input_variant'] != variant or receipt['checked_forward_calls'] != 1
                or not receipt['camera_based_tracking_and_mapping_retained']
                or type(count) is not int or count < 0 or variant == 'no_rgb' and count != 0):
            raise ValueError('actual model RGB treatment differs')
        counts.append(count)
    study = SimpleNamespace(**(vars(previous.study) | dict(ROOT=experiment.ROOT)))
    result = bind(previous.evaluate, study=study)(index, arm)
    previous.previous.save_or_read(root, 'actual_neural_rgb_treatment_v1.json',
        dict(input_variant=variant, selected_plans=len(plans), checked_forward_calls=len(counts),
            plans_with_nonzero_model_rgb=sum(c>0 for c in counts),
            model_rgb_treatment_verified=bool(plans), camera_based_tracking_and_mapping_retained=True))
    return result


if __name__ == '__main__':
    study = SimpleNamespace(**(vars(previous.study) | dict(ARMS=experiment.ARMS)))
    bind(previous.main, study=study, evaluate=evaluate)()
