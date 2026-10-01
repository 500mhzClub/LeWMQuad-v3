"""Back up the sealed_test_v2 folder off RecoveryStorage and verify it by hash (Andrew, 1 October 2026).

The seeds are recoverable only from the sealed registry inside the folder, so the folder's
files are the only copy of the set. This copies the whole folder, byte for byte, to a
separate physical disk (the workspace NVMe, outside the repository), keeping the `sealed_`
name so the AGENTS.md sealed rules apply to the copy too. It then verifies the copy without
displaying anything:
- every maze and episode file against its sha256 in the public receipt;
- the sealed registry against the receipt's registry hash;
- the remaining file (construction evidence, not in the receipt) against the source by hash;
- the file count.
Only counts and booleans are printed and written to a public backup receipt. No sealed file is
opened for its content, parsed or printed.
"""
import hashlib
import json
from pathlib import Path
import shutil

from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

DESTINATION = Path('/mnt/workspace_drive/LeWMQuad-v3_sealed_backups/sealed_test_v2')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    base = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    source = base/'sets'/'sealed_test_v2'
    receipt = json.loads((base/'sealed_test_v2_receipt.json').read_text())
    assert not DESTINATION.exists()
    assert not DESTINATION.resolve().is_relative_to(Path('/home/andrewknowles/RecoveryStorage').resolve())
    assert not DESTINATION.resolve().is_relative_to(owner.REPO.resolve())
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, DESTINATION)
    names = {f['name']: f['sha256'] for f in receipt['files']}
    registry_name = Path(receipt['sealed_registry']['path']).name
    copied = sorted(p.name for p in DESTINATION.iterdir())
    others = [n for n in copied if n not in names and n != registry_name]
    checks = dict(
        file_count=len(copied) == receipt['file_count'],
        maze_and_episode_files=all(sha(DESTINATION/n) == h for n, h in names.items()),
        sealed_registry=sha(DESTINATION/registry_name) == receipt['sealed_registry']['sha256'],
        remaining_files_match_source=bool(others) and all(sha(DESTINATION/n) == sha(source/n) for n in others),
        source_still_matches_receipt=all(sha(source/n) == h for n, h in names.items()))
    result = dict(schema='navigation_sealed_test_v2_backup_receipt.v1', backup=str(DESTINATION), device='workspace NVMe (nvme0n1), not RecoveryStorage',
                  files=len(copied), checks=checks, verified=all(checks.values()), contents_displayed=False)
    out = base/'sealed_test_v2_backup_receipt.json'
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
