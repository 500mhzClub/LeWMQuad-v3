"""Full recorded journey: exact link reuse, with original pose-output comparison."""
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.batched_patch_tracker_development import BatchedPatchVisualMotion
from lewm.cached_chain_tracker_development import CachedChainVisualMotion
from scripts import compare_full_consensus_recorded_tracker_development as source

OUTPUT = source.source.trial.run.BASE/'go2_cached_chain_recorded_tracker_v1_attempt_002'
_write = bind(source.write, OUTPUT=OUTPUT)
instances = []


class ObservedCacheTracker(CachedChainVisualMotion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        instances.append(self)


def write(name, value):
    if name == 'launch.json':
        value = value | dict(candidate='cached_image_links_with_batched_photometry',
            paired_baseline='batched_photometry',
            extra_sources={p:source.source.digest(Path(p)) for p in (
                'lewm/cached_chain_association_development.py',
                'lewm/cached_chain_tracker_development.py',
                'scripts/compare_cached_chain_recorded_tracker_development.py')})
    elif name == 'result.json':
        cache = instances[0].model.chain_association
        value = value | dict(status='CACHED_CHAIN_RECORDED_TRACKER_COMPLETE',
            cache_hits=cache.hits, cache_misses=cache.misses,
            cache_final_entries=len(cache.entries), cache_maximum_entries=cache.maximum_entries,
            writable_images_cached=False, pose_or_fit_cached=False,
            image_association_and_fit_rules_unchanged=True)
    _write(name, value)


main = bind(source.main, OUTPUT=OUTPUT, write=write,
    FullConsensusVisualMotion=ObservedCacheTracker,
    SampledPlaneChainedVisualMotion=BatchedPatchVisualMotion)


if __name__ == '__main__': main()
