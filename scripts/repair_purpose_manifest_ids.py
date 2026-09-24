from pathlib import Path
import json
p=Path('.generated/safe_local_waypoint_purpose_built_v1/state_manifest.json')
m=json.loads(p.read_text())
for i,e in enumerate(m['state_candidates']):
    e['state_id']=f'purpose-{i}'
m['states_frozen_before_branching']=True
p.write_text(json.dumps(m,indent=2))
print('repaired',len(m['state_candidates']))
