"""Count training-only temporal targets before changing the dense predictor."""
import json
from collections import Counter
from pathlib import Path

from scripts import train_go2_frozen_vjepa_native_adaptation_development as parent
from scripts import collect_go2_balanced_start_actions_development as starts

RESULT=Path('docs/go2_native_horizon_training_support_2026-09-18.json')


def main():
    assert not RESULT.exists()
    rows=parent.load_training_rows();counts={h:Counter() for h in range(1,9)}
    recordings={h:set() for h in range(1,9)};frames=set()
    missing=[]
    for row in rows:
        assert row['data_role']=='train'
        frame=row['observation_horizon_receipt']['departure_tick']
        if frame<10:continue
        directory=parent.ROOTS[row['source']]/row['trial']
        for h in range(1,9):
            if (len(row['known_commands'])<h or len(row['targets'])<h
                    or not row['targets'][h-1]['future_image_valid']):continue
            target=row['targets'][h-1]
            assert target['future_observation_index']==frame+h and target['offset_ns']==h*100_000_000
            indices=(frame-10,frame-5,frame,frame+h)
            absent=[str(directory/f'rgb_{i:04d}.png') for i in indices if not (directory/f'rgb_{i:04d}.png').is_file()]
            if absent:missing.extend(absent);continue
            frames.update(str(directory/f'rgb_{i:04d}.png') for i in indices)
            counts[h][row['source']]+=1;recordings[h].add((row['source'],row['trial']))
    collection=json.loads(starts.RESULT.read_text());assert collection['eligible']==48
    new={}
    for h in range(1,9):
        available=[]
        for case in range(48):
            root=starts.OUTPUT/f'case_{case:02d}'
            if (root/f'rgb_{10+h:04d}.png').is_file():available.append(case)
        new[h]=len(available)
    result=dict(status='COMPLETE',existing_training_offsets={str(h*100):dict(samples=sum(counts[h].values()),
                    sources=dict(counts[h]),recordings=len(recordings[h])) for h in counts},
                balanced_start_targets={str(h*100):new[h] for h in new},
                unique_existing_rgb_paths=len(frames),missing_rgb_paths=sorted(set(missing)),
                transfer_rows_used=0,new_collection=False,new_training=False,
                interpretation='The existing rows provide per-offset targets; the balanced starts stop at +500 ms and need longer native suffixes to cover +600/+700/+800 ms.',
                source_sha256=parent.digest(__file__),balanced_collection_sha256=parent.digest(starts.RESULT))
    parent.save(RESULT,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
