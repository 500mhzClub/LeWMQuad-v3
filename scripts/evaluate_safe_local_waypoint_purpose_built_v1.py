#!/usr/bin/env python3
"""Development evaluator and data-adequacy reducer for the purpose-built panel."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"

def norm_regret(best, selected, worst, eps=1e-8):
    if best is None or worst is None or best - worst <= eps:
        return None
    return max(0.0, min(1.0, (best-selected)/(best-worst)))

def reduce_state(rows, scores, safety=None):
    safe = [r for r in rows if not r["horizons"]["3"]["unsafe"]]
    if not safe:
        return {"selected_candidate": None, "admissible_count": 0, "abstention": True}
    safe_indices = [i for i, r in enumerate(rows) if not r["horizons"]["3"]["unsafe"]]
    order = sorted(safe_indices, key=lambda i: (-float(scores[i]), int(rows[i]["candidate_index"])))
    selected = rows[order[0]]
    progress = float(selected["horizons"]["3"]["progress"])
    vals = [float(r["horizons"]["3"]["progress"]) for r in safe]
    best, worst = max(vals), min(vals)
    return {"selected_candidate": int(selected["candidate_index"]),
            "admissible_count": len(safe), "selected_safety": not bool(selected["horizons"]["3"]["unsafe"]),
            "selected_progress": progress, "best_safe_progress": best,
            "absolute_regret": best-progress, "normalized_regret": norm_regret(best, progress, worst),
            "completion": bool(selected["horizons"]["3"]["completed"]), "abstention": False}

def fixture():
    rows = [{"candidate_index": i, "horizons": {"3": {"unsafe": False, "progress": float(i), "completed": i == 2}}} for i in range(4)]
    perfect = reduce_state(rows, [0, 1, 2, 4])
    reversed_ = reduce_state(rows, [4, 3, 2, 1])
    unsafe = [dict(r, horizons={"3": dict(r["horizons"]["3"])}) for r in rows]
    unsafe[3]["horizons"]["3"]["unsafe"] = True
    safety_first = reduce_state(unsafe, [0, 1, 2, 9])
    all_bad = [dict(r, horizons={"3": dict(r["horizons"]["3"], unsafe=True)}) for r in rows]
    return {"schema": "safe_local_waypoint_purpose_built_v1_fixture", "perfect": perfect,
            "reversed": reversed_, "unsafe_high_progress": safety_first,
            "all_unsafe": reduce_state(all_bad, [1, 2, 3, 4]),
            "deterministic": perfect == reduce_state(rows, [0, 1, 2, 4])}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fixture_result = fixture()
    (OUT / "evaluator_fixture.json").write_text(json.dumps(fixture_result, indent=2))
    path = OUT / "branch_labels.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    by_state = defaultdict(list)
    for r in rows: by_state[r["state_id"]].append(r)
    audit = {"rows": len(rows), "states": len(by_state), "expected_rows": 576,
             "family_counts": {}, "state_counts": {}, "safe_positive_state_fraction": None,
             "states_with_two_safe": None, "progress_nonzero": False,
             "unsafe_fraction": None, "visual_target_status": "missing_from_collector"}
    safe_pos = []; two_safe = []; family_state_flags = defaultdict(list)
    all_unsafe = []; all_progress = []
    for sid, rs in by_state.items():
        fam = rs[0]["family"]; audit["state_counts"][sid] = len(rs)
        audit["family_counts"][fam] = audit["family_counts"].get(fam, 0) + len(rs)
        safe = [r for r in rs if not r["horizons"]["3"]["unsafe"]]
        pos = [r for r in safe if r["horizons"]["3"]["progress"] > 0]
        safe_pos.append(bool(pos)); two_safe.append(len(safe) >= 2)
        family_state_flags[fam].append((bool(pos), len(safe) >= 2))
        all_unsafe.extend([r["horizons"]["3"]["unsafe"] for r in rs]); all_progress.extend([r["horizons"]["3"]["progress"] for r in rs])
    if by_state:
        audit["safe_positive_state_fraction"] = sum(safe_pos)/len(safe_pos)
        audit["states_with_two_safe"] = sum(two_safe)/len(two_safe)
    if all_unsafe: audit["unsafe_fraction"] = sum(all_unsafe)/len(all_unsafe)
    if all_progress: audit["progress_nonzero"] = (max(all_progress)-min(all_progress)) > 1e-9
    audit["per_family"] = {f: {"states": len(v), "safe_positive_fraction": sum(x[0] for x in v)/len(v),
                               "two_safe_fraction": sum(x[1] for x in v)/len(v)} for f,v in family_state_flags.items()}
    audit["adequacy_gate"] = {"required_safe_positive_fraction": .75,
                              "required_two_safe_fraction": .50,
                              "pass": bool(audit["safe_positive_state_fraction"] is not None and audit["safe_positive_state_fraction"] >= .75 and audit["states_with_two_safe"] >= .50 and audit["progress_nonzero"])}
    (OUT / "data_adequacy.json").write_text(json.dumps(audit, indent=2))
    manifest = json.loads((OUT / "state_manifest.json").read_text())
    by_family = defaultdict(list)
    for e in manifest["state_candidates"]: by_family[e["family"]].append(e["state_id"])
    split = {"fit": [], "calibration": [], "heldout": [], "policy": "8/2/2 states per family, frozen manifest order"}
    for fam, ids in by_family.items():
        ids = sorted(ids, key=lambda x: int(x.split('-')[-1]))
        split["fit"] += ids[:8]; split["calibration"] += ids[8:10]; split["heldout"] += ids[10:12]
    (OUT / "split.json").write_text(json.dumps(split, indent=2))
    print(json.dumps({"fixture": fixture_result, "audit": audit}, indent=2))

if __name__ == "__main__": main()
