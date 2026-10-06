"""Unmarked twin of the stage-2 held-out evaluation set (development; Andrew, 5 October 2026: the unmarked control is run
offline at the forecast level).

The items are exactly the stage-2 cache's `eval_patch` items (scripts/build_go2_stage2_feature_cache_development.py
`collect`; same frames, tapes, command histories, targets, horizons and edge bins), but their frames are read from the
unmarked re-renders (`stage2_unmarked_rerenders/<assignment>/ego_frames`, scripts/render_go2_stage2_unmarked_frames_development.py),
whose physics is asserted identical to the marked recordings. Encoding is the decoder fix's `main()` unchanged.

So the marked and unmarked caches form an exact pair: per context, only the floor tint in the RGB differs.

Output: `<capability root>/stage2_unmarked_eval_cache_v1`.
"""
import json
from pathlib import Path

from scripts import build_go2_stage2_feature_cache_development as stage2

v1 = stage2.v1
OUT = v1.BASE/'stage2_unmarked_eval_cache_v1'
RERENDERS = v1.BASE/'stage2_unmarked_rerenders'
STAGE2_COLLECT = stage2.collect


def collect():
    items, sources = STAGE2_COLLECT()
    keep = [it for it in items if it['set'] == 'eval_patch']
    unmarked = {}
    for it in keep:
        run = Path(it['source'])
        verification = RERENDERS/run.name/'rerender_verification.json'
        if not json.loads(verification.read_text())['passed']:
            raise ValueError(f'unverified unmarked re-render: {run.name}')
        if it['source'] not in unmarked:
            unmarked[it['source']] = v1.training_source(str(run), str(RERENDERS/run.name))
    return keep, unmarked


if __name__ == '__main__':
    v1.OUT, v1.collect = OUT, collect
    v1.main()
