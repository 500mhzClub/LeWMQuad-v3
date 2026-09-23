"""Freeze only the explicitly approved V4.2 delta from V4.1."""
import copy,hashlib,json
from pathlib import Path
BASE=Path('docs/go2_decision_headroom_protocol_v41_2026-09-23.json')
EXPECTED='e7735ae9d98e6210d9e17f8301b5d6817826ce0b2a87d1971e5488cfa68fe190'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==EXPECTED
old=json.loads(BASE.read_text());new=copy.deepcopy(old)
new['schema']='go2_decision_headroom_protocol.v4.2'
new['authority']='Explicit user approval of V4.2 = identified V4.1 plus output converter, its implementation bindings, and 72-hour GPU-owner allowance; exact approval text bound separately.'
new['entrypoint']='scripts/run_go2_headroom_v42_development.py --approval docs/go2_decision_headroom_v42_approval_2026-09-23.json'
new['historical_rerender_execution']['entrypoint']='scripts/check_go2_headroom_historical_rerender_v42_development.py'
new['execution_caps']['compute_caps']['gpu_owner_wall_seconds']=72*3600
new['output_converter']=dict(path='lewm/decision_headroom_json_v42_development.py',rule='NumPy booleans, integers and floats via .item(); arrays via .tolist(); all other types unchanged',writes='All JSON and JSONL audit output, including inherited collection/branch writers; complete-file or record readback with exact JSON structural/value equality',readback_mismatch='mandatory stop',runtime_writer_failure='mandatory stop; never a quantity-local exclusion',prelaunch='Complete V4.2 owner on six already-qualified states in separate root, using retained traces when available; no comparative results retained as science; all written outputs read back and schema validated; full analysis, panel assembly and memo inputs',prelaunch_additional_numpy_serialization_failures='Converter-only handling pre-authorized, each failure logged; any other defect stops for disposition',implementation_check_flag='--implementation-check',implementation_check_output_root=str(Path(new['execution_caps']['output_root']).parent/'go2_headroom_v42_output_check_attempt_001'),implementation_caps='Unchanged execution caps; retained-trace check advances zero physics',schema='Explicit required audit envelopes plus exact in-memory JSON structure and value checks; binary retained outputs hash-checked and format-validated',other_types='Ordinary Python JSON semantics are preserved: tuples serialize as arrays and legal scalar keys use JSON key spellings; no rounding, masking, imputation, controller or scientific calculation changes')
files=['lewm/decision_headroom_json_v42_development.py',
 'scripts/run_go2_headroom_v42_development.py','scripts/run_go2_headroom_v42_source_development.py',
 'scripts/run_go2_headroom_v42_branches_development.py','scripts/read_go2_headroom_v42_development.py',
 'scripts/check_go2_headroom_v42_owner_development.py','scripts/check_go2_headroom_historical_rerender_v42_development.py',
 'scripts/freeze_go2_headroom_v42_development.py']
for f in files:
 p=Path(f);new['frozen_bindings'][f]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
allowed={('schema',),('authority',),('entrypoint',),('historical_rerender_execution','entrypoint'),('execution_caps','compute_caps','gpu_owner_wall_seconds'),('output_converter',)}|{('frozen_bindings',p) for p in files}
diffs=[]
def compare(a,b,path=()):
 if a==b:return
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(set(a)|set(b)):
   if k not in a or k not in b:diffs.append(dict(path=list(path+(k,)),before=a.get(k),after=b.get(k)))
   else:compare(a[k],b[k],path+(k,))
 else:diffs.append(dict(path=list(path),before=a,after=b))
compare(old,new)
assert all(tuple(d['path']) in allowed for d in diffs)
for k,v in old['frozen_bindings'].items():assert new['frozen_bindings'][k]==v
caps=copy.deepcopy(new['execution_caps']);caps['compute_caps']['gpu_owner_wall_seconds']=old['execution_caps']['compute_caps']['gpu_owner_wall_seconds'];assert caps==old['execution_caps']
p=Path('docs/go2_decision_headroom_protocol_v42_2026-09-23.json');p.write_text(json.dumps(new,indent=2)+'\n');digest=hashlib.sha256(p.read_bytes()).hexdigest()
p.with_suffix('.sha256').write_text(digest+'  '+p.name+'\n')
Path('docs/go2_decision_headroom_v42_diff_check_2026-09-23.json').write_text(json.dumps(dict(status='PASS',baseline_sha256=EXPECTED,protocol_sha256=digest,allowed_categories=['version metadata','converter and implementation bindings','GPU-owner allowance'],differences=diffs,unlisted_differences=[],all_other_caps_identical=True,all_prior_bindings_preserved=True),indent=2)+'\n')
print(digest)
