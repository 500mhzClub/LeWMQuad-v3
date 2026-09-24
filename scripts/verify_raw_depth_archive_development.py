"""Measure compact archive storage and compare every reconstructed public packet."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from PIL import Image
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.in_memory_paired_camera_session_development import persist_camera_pair, save_depth_archive
from scripts.raw_depth_archive_development import SCHEMA

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def equal(a,b):
    if type(a) is not type(b):raise AssertionError('packet value types differ')
    if isinstance(a,np.ndarray):
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise AssertionError('packet arrays differ')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('packet fields differ')
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError('packet sequence length differs')
        for x,y in zip(a,b,strict=True):equal(x,y)
    elif a!=b:raise AssertionError('packet value differs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary development basename required')
    root=BASE/args.root_name;source=root/'native';output=root/'raw_depth_archive_verification_v1'
    output.mkdir();old=PublicReplay(source)
    metadata=json.loads((source/'in_memory_camera_observations.json').read_text());rows=metadata['frames']
    started=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='lewm_raw_depth_') as directory:
        directory=Path(directory)
        for name in ('policy_histories.npz','fast_gyro_histories.npz'):shutil.copyfile(source/name,directory/name)
        def convert(row):
            frame=row['frame'];p,d,fast,rgb,a,now=old.packet(frame);images=[];old_bytes=0
            for label in ('primary','auxiliary'):
                name=f'{label}_depth_{frame:04d}.npz';old_bytes+=(source/name).stat().st_size
                with np.load(source/name,allow_pickle=False) as archive:native=archive['native_optical_depth_m']
                rgb_name=f'rgb_{frame:04d}.png' if label=='primary' else f'auxiliary_rgb_{frame:04d}.png'
                with Image.open(source/rgb_name) as image:pixels=np.array(image)
                images.append((pixels,native))
            saved=persist_camera_pair(row|dict(images=images,depth=d,auxiliary_depth=a),directory,native_depth_only=True)
            new_bytes=sum((directory/f'{label}_depth_{frame:04d}.npz').stat().st_size for label in ('primary','auxiliary'))
            return saved,old_bytes,new_bytes
        with ThreadPoolExecutor(max_workers=4) as executor:converted=list(executor.map(convert,rows))
        compact=metadata|dict(schema=SCHEMA,frames=[v[0] for v in converted])
        (directory/'in_memory_camera_observations.json').write_text(json.dumps(compact))
        new=PublicReplay(directory)
        def compare(frame):equal(old.packet(frame),new.packet(frame));return frame
        with ThreadPoolExecutor(max_workers=4) as executor:checked=list(executor.map(compare,range(len(rows))))
        # Exercise detection of both altered derived-packet witnesses and raw pixels.
        witness=new.depth_witnesses[0]['primary'];original=witness['derived_packet_sha256']
        witness['derived_packet_sha256']='0'*64
        try:new.packet(0)
        except ValueError:changed_witness_rejected=True
        else:raise AssertionError('altered packet witness admitted')
        witness['derived_packet_sha256']=original
        leaf=directory/'primary_depth_0000.npz'
        with np.load(leaf,allow_pickle=False) as archive:native=archive['native_optical_depth_m'].copy()
        native[0,0]=np.nextafter(native[0,0],np.float32(0.))
        leaf.unlink();save_depth_archive(leaf,native_optical_depth_m=native)
        try:new.packet(0)
        except ValueError:changed_raw_rejected=True
        else:raise AssertionError('altered raw pixels admitted')
        old_bytes=sum(v[1] for v in converted);new_bytes=sum(v[2] for v in converted)
        report=dict(frames=len(checked),all_public_packets_byte_equal=True,
            changed_witness_rejected=changed_witness_rejected,changed_raw_rejected=changed_raw_rejected,
            original_depth_bytes=old_bytes,raw_only_depth_bytes=new_bytes,
            depth_bytes_saved=old_bytes-new_bytes,depth_saving_fraction=1-new_bytes/old_bytes,
            elapsed_s=time.perf_counter()-started,workers=4,native_state_used=False,
            original_artifacts_changed=False,temporary_regenerated_copies_retained=False)
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
