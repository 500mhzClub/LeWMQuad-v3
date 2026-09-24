import ast
from pathlib import Path


def test_privileged_inspection_preserves_all_checks_and_drops_privileges_before_mutation():
    root = Path(__file__).resolve().parents[2]
    old = (root/'scripts/retire_go2_reviewed_geometry_cache_v1.py').read_text()
    expected = old.replace('Apply an exact cache-retirement proposal only after explicit user authorization.',
        'Inspect protected process descriptors as root, then retire approved cache as its owner.')
    expected = expected.replace("    if not __debug__: raise ValueError('assertions required')",
        "    if not __debug__: raise ValueError('assertions required')\n"
        "    if os.getuid() != 0 or os.geteuid() != 0:\n"
        "        raise ValueError('administrator access required only for process inspection')\n"
        "    target_uid = target_gid = 1000")
    expected = expected.replace('if process.uids().real != os.getuid(): continue',
        'if process.uids().real != target_uid: continue')
    expected = expected.replace("    entries = proposal['candidate_metadata']; keep = proposal['retained_current_geometry_keys']",
        "    os.setgroups([])\n    os.setgid(target_gid)\n    os.setuid(target_uid)\n"
        "    if os.getresuid() != (target_uid, target_uid, target_uid) or os.getresgid() != (target_gid, target_gid, target_gid):\n"
        "        raise ValueError('permanent return to cache owner required before mutation')\n"
        "    entries = proposal['candidate_metadata']; keep = proposal['retained_current_geometry_keys']")
    inspection = "    inspection = Path(proposal['inspection_root']); validate_root(inspection)\n    verify_artifacts(inspection, proposal['inspection_artifact_sha256'])\n"
    expected = expected.replace(inspection, '')
    expected = expected.replace("    entries = proposal['candidate_metadata']; keep = proposal['retained_current_geometry_keys']",
        inspection + "    entries = proposal['candidate_metadata']; keep = proposal['retained_current_geometry_keys']")
    actual = (root/'scripts/retire_go2_reviewed_geometry_cache_privileged_v1.py').read_text()
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    main = next(n for n in ast.parse(actual).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(main) if isinstance(n, ast.Call)]
    drop_line = next(line for line, name in calls if name == 'os.setuid')
    for name in ('validate', 'validate_root', 'verify_artifacts', 'create_output', 'write_json', 'retire'):
        assert all(line > drop_line for line, called in calls if called == name)
