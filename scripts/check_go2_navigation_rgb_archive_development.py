"""Lossless RGB archive check on a retained development episode; no physics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from PIL import Image
import numpy as np

from lewm import decision_headroom_json_v42_development as output
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, save, sha


def check(source):
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root'])
    assert source.resolve().is_relative_to((base/'runs').resolve())
    assert json.loads((source/'episode.json').read_text())['role']=='dev_tune'
    root=base/'rgb_archive_check_attempt001';root.mkdir(exist_ok=False);output.install(base)
    budget=Budget(base,protocol);budget.admit_persist(256*1024**2)
    rows=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames']
    save(root/'config.json',dict(source=str(source),source_config_sha256=sha(source/'config.json'),
        frames=len(rows),codec='libx264rgb',crf=0,preset='fast',pixel_format='rgb24',
        purpose='Output-preserving prospective recording check; originals preserved',
        script_sha256=sha(__file__),physics=False))
    started=time.monotonic();results={}
    try:
        for label,prefix in [('primary','rgb'),('auxiliary','auxiliary_rgb')]:
            path=root/f'{label}.mkv';log=root/f'{label}_encode.log'
            with log.open('x') as stream:
                subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','2','-framerate','10',
                    '-i',str(source/f'native/{prefix}_%04d.png'),'-frames:v',str(len(rows)),
                    '-an','-c:v','libx264rgb','-threads','2','-crf','0','-preset','fast',
                    '-pix_fmt','rgb24',str(path)],stderr=stream,check=True)
            process=subprocess.Popen(['ffmpeg','-nostdin','-v','error','-threads','2','-i',str(path),
                '-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],stdout=subprocess.PIPE)
            matches=0;original_bytes=0
            try:
                for row in rows:
                    budget.check()
                    data=process.stdout.read(640*480*3)
                    if len(data)!=640*480*3:raise ValueError('short decoded RGB frame')
                    if hashlib.sha256(data).hexdigest()!=row['pixel_sha256'][label]['rgb_sha256']:
                        raise ValueError(f'RGB archive mismatch at {label}/{row["frame"]}')
                    original_bytes+=(source/f'native/{prefix}_{row["frame"]:04d}.png').stat().st_size
                    matches+=1
                if process.stdout.read(1):raise ValueError('unexpected extra RGB archive frame')
                if process.wait():raise ValueError('RGB archive decoder failed')
            finally:
                process.stdout.close()
                if process.poll() is None:process.terminate();process.wait()
            results[label]=dict(bitwise_matches=matches,original_png_bytes=original_bytes,
                archive_bytes=path.stat().st_size,archive_sha256=sha(path))
        save(root/'result.json',dict(passed=True,results=results,wall_s=time.monotonic()-started,
            sensor_values_changed=False,source_files_changed=False,decisions_recomputed=False))
    except BaseException as exc:
        save(root/'failure.json',dict(reason=repr(exc),automatic_retry=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    check(parser.parse_args().source)
