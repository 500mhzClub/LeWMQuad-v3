"""Enforce the recorded simulator/model/driver environment before an episode."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def verify_environment(path):
    import torch
    pin=json.loads(path.read_text())
    current=dict(python=sys.version,kernel=platform.release(),machine=platform.machine(),
        torch=str(torch.__version__),rocm=str(torch.version.hip))
    for key,value in current.items():
        if value!=pin[key]:raise RuntimeError('Pinned environment changed: '+key)
    distributions=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())
    expected=sorted((d['name'],d['version']) for d in pin['distributions'])
    if distributions!=expected:raise RuntimeError('Pinned Python distribution versions changed')
    genesis=pin['genesis']
    if hashlib.sha256(Path(genesis['module']).read_bytes()).hexdigest()!=genesis['module_sha256']:
        raise RuntimeError('Pinned simulator source changed')
    if Path('/sys/module/amdgpu/srcversion').read_text().strip()!=pin['loaded_amdgpu_srcversion']:
        raise RuntimeError('Loaded GPU driver changed')
    names=[line.split('\t')[0] for line in pin['driver_packages']['stdout'].splitlines()]
    versions=subprocess.run(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n',*names],
        capture_output=True,text=True,check=True).stdout
    if sorted(versions.splitlines())!=sorted(pin['driver_packages']['stdout'].splitlines()):
        raise RuntimeError('GPU/ROCm system packages changed')
    for key,value in pin['render_environment'].items():
        if os.environ.get(key)!=value:raise RuntimeError('Pinned render environment changed: '+key)
    repo=Path(__file__).resolve().parents[1]
    for name,binding in pin['model_and_harness_code_bindings'].items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=binding['sha256']:
            raise RuntimeError('Frozen predecessor/model implementation changed: '+name)
    return dict(pin_path=str(path),pin_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        verified=True,software=current,devices=[dict(index=i,name=torch.cuda.get_device_name(i),
        capacity_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])
