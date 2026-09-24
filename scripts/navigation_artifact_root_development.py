"""Narrow new-development output authority; ordinary source guards unchanged.

No relocation, source export, directory discovery or predecessor data access.
Only an explicitly named fresh attempt under this fixed owned root may be
created/read/verified. Protected path components and every symlink are rejected.
"""
import os
from pathlib import Path
import re
from scripts.run_go2_successive_choice_maze_development_v1 import digest

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def _ordinary(path):
    if any(x in ('sealed','sealed_test.json') or x.startswith('sealed_') for x in path.parts):
        raise ValueError('protected artifact path forbidden')


def validate_root(output,*,must_exist=True):
    output=Path(output);_ordinary(output)
    if (not output.is_absolute() or output.parent!=BASE
            or re.fullmatch(r'go2_[a-z0-9_]+_attempt_[0-9]{3}',output.name) is None
            or BASE.resolve()!=BASE or output.resolve()!=output):
        raise ValueError('exact nonsymlink development attempt root required')
    for p in (BASE,output):
        if p.exists() and (not p.is_dir() or p.stat().st_uid!=os.getuid()):raise ValueError('owned artifact directory required')
    if must_exist and not output.is_dir():raise ValueError('existing explicit output required')
    return output


def create_output(output):
    output=validate_root(output,must_exist=False)
    if output.exists() or output.is_symlink():raise ValueError('exclusive fresh artifact attempt')
    BASE.mkdir(exist_ok=True)
    validate_root(output,must_exist=False);output.mkdir()
    return validate_root(output)


def artifact_path(output,name):
    output=validate_root(output);p=Path(name);_ordinary(p)
    if (type(name) is not str or not name or p.is_absolute() or str(p)!=name
            or any(x in ('.','..') for x in p.parts)):
        raise ValueError('canonical ordinary relative artifact path required')
    target=output/p
    if target.resolve()!=target or not target.is_file() or target.stat().st_uid!=os.getuid():
        raise ValueError('existing owned nonsymlink artifact required')
    return target


def verify_artifacts(output,bindings):
    validate_root(output)
    if not isinstance(bindings,dict):raise ValueError('explicit artifact hash map required')
    for name,h in bindings.items():
        if not isinstance(h,str) or re.fullmatch('[0-9a-f]{64}',h) is None:raise ValueError('exact SHA256 binding required')
        if digest(artifact_path(output,name))!=h:raise ValueError('artifact identity changed: '+name)
