"""Side-by-side 2x2 composite of replay-verified capability videos sharing one episode.

Inputs are published per-controller videos (each already replay-verified); the
composite only rescales and pads (shorter missions hold their final frame).
"""
import argparse
import json
from pathlib import Path
import subprocess

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner


def duration(path):
    probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)],
                           capture_output=True, text=True, check=True)
    return float(json.loads(probe.stdout)['format']['duration'])


def compose(inputs, destination, speed=1):
    longest = max(duration(p) for p in inputs)
    filters = []
    for i, path in enumerate(inputs):
        pad = longest-duration(path)
        filters.append(f'[{i}:v]scale=960:540,tpad=stop_mode=clone:stop_duration={pad:.3f}[v{i}]')
    stack = ''.join(f'[v{i}]' for i in range(4))
    tail = ',setpts=PTS/4' if speed == 4 else ''
    label = (",drawbox=x=720:y=500:w=480:h=80:color=black@0.8:t=fill,"
             "drawtext=text='4x SIMULATED TIME':x=770:y=522:fontsize=40:fontcolor=white") if speed == 4 else ''
    filters.append(f'{stack}xstack=inputs=4:layout=0_0|960_0|0_540|960_540{tail}{label}[out]')
    command = ['ffmpeg', '-nostdin', '-v', 'error']
    for path in inputs:
        command += ['-i', str(path)]
    command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-r', '30', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True)
    return longest


def main(roots, label):
    protocol = json.loads(owner.PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    output.install(base)
    metadata = [json.loads((r/'metadata.json').read_text()) for r in roots]
    assert [m['controller'] for m in metadata] == ['C1', 'C2', 'C3', 'C4']
    assert len({m['episode'] for m in metadata}) == 1 and all(m['status'] == 'CAPABILITY_VIDEO_REPLAY_VERIFIED' for m in metadata)
    videos = [r/next(k for k in m['outputs'] if not k.endswith('_4x.mp4')) for r, m in zip(roots, metadata)]
    for video, m in zip(videos, metadata):
        assert owner.sha(video) == m['outputs'][video.name]
    root = base/'videos'/label
    root.mkdir(exist_ok=False)
    stem = f'composite_C1_C2_C3_C4_val{metadata[0]["episode"].replace("/", "_ep")}'
    longest = compose(videos, root/f'{stem}.mp4')
    outputs = {f'{stem}.mp4': owner.sha(root/f'{stem}.mp4')}
    if longest > 120:
        compose(videos, root/f'{stem}_4x.mp4', speed=4)
        outputs[f'{stem}_4x.mp4'] = owner.sha(root/f'{stem}_4x.mp4')
    owner.save(root/'metadata.json', dict(status='COMPOSITE_OF_REPLAY_VERIFIED_VIDEOS', episode=metadata[0]['episode'],
        layout='C1 top-left, C2 top-right, C3 bottom-left, C4 bottom-right; shorter missions hold their final frame',
        inputs={str(v): owner.sha(v) for v in videos}, input_metadata_sha256={str(r): owner.sha(r/'metadata.json') for r in roots},
        outputs=outputs, longest_s=longest, composer_sha256=owner.sha(__file__)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--roots', type=Path, nargs=4, required=True)
    p.add_argument('--label', required=True)
    args = p.parse_args()
    main(args.roots, args.label)
