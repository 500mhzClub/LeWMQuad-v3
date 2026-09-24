"""Full recorded-tracker comparison after the actual-chain batching benchmark."""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.full_consensus_tracker_development import FullConsensusVisualMotion
from lewm.batched_patch_tracker_development import BatchedPatchVisualMotion
from scripts import compare_full_consensus_recorded_tracker_development as source

OUTPUT = source.source.trial.run.BASE/'go2_batched_patch_recorded_tracker_v1_attempt_001'
_write = bind(source.write, OUTPUT=OUTPUT)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(candidate='batched_photometry_with_full_consensus',
            paired_baseline='scalar_photometry_with_full_consensus', auxiliary_features_deferred=False,
            extra_sources={p:source.source.digest(Path(p)) for p in (
                'lewm/batched_patch_agreement_development.py',
                'lewm/batched_patch_tracker_development.py',
                'scripts/compare_batched_patch_recorded_tracker_development.py')})
    elif name == 'result.json':
        value = value | dict(status='BATCHED_PATCH_RECORDED_TRACKER_COMPLETE',
            photometry_batched=True, auxiliary_features_deferred=False,
            image_association_and_fit_rules_unchanged=True)
    _write(name, value)


main = bind(source.main, OUTPUT=OUTPUT, write=write,
    FullConsensusVisualMotion=BatchedPatchVisualMotion,
    SampledPlaneChainedVisualMotion=FullConsensusVisualMotion)


if __name__ == '__main__': main()
