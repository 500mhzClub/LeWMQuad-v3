"""One lossless NumPy JSON converter and read-after-write audit boundary.

Only NumPy bool/integer/float/array types are converted. Ordinary JSON container
traversal preserves container types. Readback compares JSON sequence/object
semantics (tuples are JSON arrays; legal non-string keys use JSON's key spelling).
Writer failures derive from BaseException so state-local handlers cannot swallow
the user's mandatory stop. Installation is restricted to the audit output root.
"""
import builtins
import io
import json
import math
import os
from pathlib import Path
import threading
import numpy as np

_dump, _dumps, _open, _io_open = json.dump, json.dumps, builtins.open, io.open
_root = None
_lock = threading.RLock()
_counts = {}
_files = {}


class OutputFailure(BaseException):
    pass


def converter(value):
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        with _lock: _counts[type(value).__name__] = _counts.get(type(value).__name__, 0)+1
        return value.item()
    if isinstance(value, np.ndarray):
        with _lock: _counts['ndarray'] = _counts.get('ndarray', 0)+1
        return value.tolist()
    return value


def converted(value):
    result = converter(value)
    if result is not value: return converted(result)
    if isinstance(value, dict): return {converter(k):converted(v) for k,v in value.items()}
    if isinstance(value, list): return [converted(v) for v in value]
    if isinstance(value, tuple): return tuple(converted(v) for v in value)
    return value


def key_string(key):
    return key if isinstance(key,str) else _dumps(key)


def equal(actual, expected):
    if isinstance(expected,dict):
        keys=[key_string(k) for k in expected]
        return isinstance(actual,dict) and len(keys)==len(set(keys)) and set(actual)==set(keys) and all(equal(actual[key_string(k)],v) for k,v in expected.items())
    if isinstance(expected,(tuple,list)):
        return isinstance(actual,list) and len(actual)==len(expected) and all(equal(a,b) for a,b in zip(actual,expected))
    if type(actual) is not type(expected): return False
    if isinstance(expected,float) and math.isnan(expected): return math.isnan(actual)
    return actual==expected


def dumps(value,*args,**kwargs):
    try:
        memory=converted(value)
        result=_dumps(memory,*args,**kwargs)
        if not equal(json.loads(result),memory): raise ValueError('in-memory JSON round-trip mismatch')
        return result
    except OutputFailure: raise
    except BaseException as exc: raise OutputFailure('JSON conversion/readback: '+repr(exc)) from exc


def dump(value, stream, *args, **kwargs):
    text=dumps(value,*args,**kwargs)
    try:
        stream.write(text)
        stream.flush()
    except OutputFailure: raise
    except BaseException as exc: raise OutputFailure('JSON writer: '+repr(exc)) from exc


def schema(path, value):
    """Validate explicit audit envelopes; other files retain their input schema."""
    name=Path(path).name
    required={
        'audit_v4.json':('objective','sampling','source_input_valid','physics_valid','rgb_valid','safety','reference','costs','rows','filter_audit','localisation','secondary_reference','secondary_costs'),
        'articulated_v4.json':('hard','operating','contact','complete','per_step_primitive_separation_lower_m','primitive_ids'),
        'restoration.json':('source_trace_complete','comparisons'),
        'source_input_fidelity.json':('status','failures','frame'),
        'analysis_v41.json':('cells','layout_clustered','primary_family_size','reference_coverage_by_clearance'),
        'analysis_v42.json':('cells','layout_clustered','primary_family_size','reference_coverage_by_clearance'),
        'branch_panel_v42.json':('schema','implementation_only','states'),
        'memo_inputs_v42.json':('schema','implementation_only','primary_quantities','layout_clustered'),
    }.get(name,())
    if required and (not isinstance(value,dict) or any(k not in value for k in required)):
        raise ValueError('missing audit schema fields: '+name)
    if name=='snapshots.json' and (not isinstance(value,list) or any('frame' not in s for s in value)):
        raise ValueError('snapshot list schema')
    if name=='audit_v4.json':
        assert len(value['safety'])==6
        assert all(type(value[k]) is bool for k in ('source_input_valid','physics_valid','rgb_valid'))
        for row in value['filter_audit'].values():
            for criterion in ('hard','operating'): assert len(row[criterion]['candidates'])==5
    if name=='articulated_v4.json':
        assert type(value['complete']) is bool and len(value['primitive_ids'])==27
        assert all(len(row)==27 for row in value['per_step_primitive_separation_lower_m'])
    if name.startswith('analysis_v4') and name.endswith('.json'):
        assert value['primary_family_size']==13


class CheckedFile:
    def __init__(self, stream): self.stream=stream
    def __getattr__(self,name): return getattr(self.stream,name)
    def __enter__(self): return self
    def __exit__(self,*args):
        try:
            if args[0] is None: self.validate()
        finally: self.stream.close()
    def write(self,text):
        try:
            start=self.stream.tell();n=self.stream.write(text);self.stream.flush();end=self.stream.tell()
            with _open(self.stream.name,'rb') as reader:
                reader.seek(start);actual=reader.read(end-start).decode(self.stream.encoding or 'utf-8')
            if actual!=text: raise ValueError('written bytes differ from in-memory text')
            if str(self.stream.name).endswith('.jsonl') and text.strip():
                for line in actual.splitlines(): schema(self.stream.name,json.loads(line))
            with _lock: _files[str(self.stream.name)]=dict(readback_equal=True,schema_validated=False)
            return n
        except OutputFailure: raise
        except BaseException as exc: raise OutputFailure(str(self.stream.name)+': '+repr(exc)) from exc
    def validate(self):
        try:
            self.stream.flush()
            with _open(self.stream.name) as reader:
                if str(self.stream.name).endswith('.jsonl'):
                    for line in reader:
                        if line.strip(): schema(self.stream.name,json.loads(line))
                else: schema(self.stream.name,json.load(reader))
            with _lock: _files[str(self.stream.name)]=dict(readback_equal=True,schema_validated=True)
        except BaseException as exc: raise OutputFailure(str(self.stream.name)+': '+repr(exc)) from exc
    def close(self):
        try: self.validate()
        finally: self.stream.close()


def checked_open(original, file, *args, **kwargs):
    stream=original(file,*args,**kwargs)
    mode=kwargs.get('mode',args[0] if args else 'r')
    if _root is not None and isinstance(file,(str,os.PathLike)) and any(c in mode for c in 'wax+') and 'b' not in mode:
        path=Path(file).absolute()
        if path.suffix in ('.json','.jsonl') and path.is_relative_to(_root): return CheckedFile(stream)
    return stream


def install(root):
    global _root
    _root=Path(root).absolute()
    os.environ['LEWM_HEADROOM_JSON_ROOT']=str(_root)
    json.dump=dump;json.dumps=dumps
    builtins.open=lambda file,*a,**k:checked_open(_open,file,*a,**k)
    io.open=lambda file,*a,**k:checked_open(_io_open,file,*a,**k)


def receipt():
    with _lock: return dict(conversions=dict(_counts),files=dict(_files),conversion_rule='NumPy scalar .item(); ndarray .tolist(); other types unchanged')


if os.environ.get('LEWM_HEADROOM_JSON_ROOT'): install(os.environ['LEWM_HEADROOM_JSON_ROOT'])
