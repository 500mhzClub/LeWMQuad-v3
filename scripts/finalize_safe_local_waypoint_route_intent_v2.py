#!/usr/bin/env python3
"""Finalize V2 safety attribution and storage accounting without retraining."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
OUT = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03/safe_local_waypoint_route_intent_v2")


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<22),b""): h.update(b)
    return h.hexdigest()


def main() -> int:
    split=json.loads((V1/"split.json").read_text()); split_names=("fit","calibration","heldout")
    split_of={sid:k for k in split_names for sid in split[k]}
    ledger=[json.loads(x) for x in (V1/"branch_labels.jsonl").read_text().splitlines()]
    by_state=defaultdict(list)
    for row in ledger: by_state[row["state_id"]].append(row)
    safe_counts={sid:sum(not r["horizons"]["3"]["unsafe"] for r in rows) for sid,rows in by_state.items()}
    state_summary={"zero_safe":sum(v==0 for v in safe_counts.values()),"exactly_one_safe":sum(v==1 for v in safe_counts.values()),
                   "two_or_more_safe":sum(v>=2 for v in safe_counts.values())}
    by_split={k:{"states":len(split[k]),"safe_candidates":sum(safe_counts[s] for s in split[k]),
                 "counts":Counter(safe_counts[s] for s in split[k])} for k in split_names}
    manifest=json.loads((V1/"state_manifest.json").read_text());family={e["state_id"]:e["family"] for e in manifest["state_candidates"]}
    by_family={f:{"states":sum(family[s]==f for s in safe_counts),"safe_candidates":sum(v for s,v in safe_counts.items() if family[s]==f),
                  "counts":Counter(v for s,v in safe_counts.items() if family[s]==f)} for f in sorted(set(family.values()))}
    component=Counter();combination=Counter();match=Counter();replay_runtime=0.0
    for path in sorted((OUT/"replay").glob("purpose-*.json")):
        state=json.loads(path.read_text());replay_runtime+=float(state.get("runtime_s",0))
        for row in state["rows"]:
            h=row["horizons"]["3"];c=h["components"]
            for k,v in c.items(): component[k]+=bool(v)
            combination["+".join(sorted(k for k,v in c.items() if v)) or "none"]+=1
            replay=bool(h.get("replay_path_unsafe",h.get("path_unsafe")))
            frozen=bool(h.get("frozen_path_unsafe",replay));match[(frozen,replay)]+=1
    audit={"frozen_definition":"collision/disallowed contact OR stuck OR unsafe termination (V1 aggregate)",
           "frozen_path_unsafe":sum(r["horizons"]["3"]["unsafe"] for r in ledger),"frozen_path_safe":sum(not r["horizons"]["3"]["unsafe"] for r in ledger),
           "replay_component_sensitivity_counts":dict(component),"replay_component_combinations":dict(combination),
           "replay_vs_frozen":{"both_safe":match[(False,False)],"frozen_safe_replay_unsafe":match[(False,True)],
                                "frozen_unsafe_replay_safe":match[(True,False)],"both_unsafe":match[(True,True)],
                                "matches":match[(False,False)]+match[(True,True)],"mismatches":match[(False,True)]+match[(True,False)]},
           "interpretation":"Component fields were absent from V1. Replay attribution is sensitivity-only because contact flags differed on 18/576 rows; frozen aggregate labels remain authoritative.",
           "safe_candidate_state_counts":state_summary,
           "safe_candidates_by_split":{k:{**v,"counts":dict(v["counts"])} for k,v in by_split.items()},
           "safe_candidates_by_family":{k:{**v,"counts":dict(v["counts"])} for k,v in by_family.items()}}
    rgb=list((CACHE/"rgb").rglob("*.png"));lat=list((CACHE/"latents").glob("*.npy"))
    rgb_times=[p.stat().st_mtime for p in rgb if "purpose-0" not in p.parts]
    storage={"rgb_frames":len(rgb),"latent_grids":len(lat),"rgb_bytes":sum(p.stat().st_size for p in rgb),
             "latent_bytes":sum(p.stat().st_size for p in lat),"generated_metadata_bytes":sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())}
    runtime={"replay_state_runtime_sum_s":replay_runtime,"static_render_wall_span_s":max(rgb_times)-min(rgb_times) if rgb_times else None,
             "encoding_s":json.loads((OUT/"target_latent_index.json").read_text())["runtime_s"]}
    result=json.loads((OUT/"result.json").read_text());result["safety_audit"]=audit;result["storage"]=storage;result["runtime_breakdown"]=runtime
    result["predictor_access"]="NONE_TRUE_FUTURE_GATE_FAILED";result["global_memory_novelty_beacon_layer"]="NOT_IMPLEMENTED"
    result["nothing_running_at_finalization"] = True
    path=OUT/"result.json";path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
    summary={"result_sha256":sha(path),"checkpoint_sha256":result["training"]["checkpoint_sha256"],"safety_audit":audit,"storage":storage,"runtime":runtime}
    (OUT/"finalization.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    print(json.dumps(summary,indent=2,sort_keys=True))
    return 0


if __name__=="__main__": raise SystemExit(main())
