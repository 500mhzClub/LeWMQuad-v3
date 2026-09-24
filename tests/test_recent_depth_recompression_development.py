from io import BytesIO
import json
import zipfile

import numpy as np
import pytest

from scripts.recompress_recent_depth_archives_development import prepare, replace, sha


def source_archive(path):
    # Include signed zero, infinities and distinct NaN payload bits.
    array = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
        0x7fc00001, 0x7fc00002, 0x3f800000], dtype=np.uint32).view(np.float32)
    buffer = BytesIO(); np.save(buffer, array, allow_pickle=False)
    member = buffer.getvalue()
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('native_optical_depth_m.npy', member)
    return array, member


def test_lossless_replacement_preserves_nan_bits_and_hash_history(tmp_path):
    path = tmp_path / 'primary_depth_0000.npz'
    array, member = source_archive(path)
    original = path.read_bytes()
    compressed, receipt = prepare(path)
    assert path.read_bytes() == original
    with (tmp_path / 'journal.jsonl').open('x') as journal:
        replace(path, compressed, receipt, journal)
    with zipfile.ZipFile(path) as archive:
        assert archive.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert archive.read('native_optical_depth_m.npy') == member
    with np.load(path, allow_pickle=False) as archive:
        assert archive['native_optical_depth_m'].tobytes() == array.tobytes()
    rows = [json.loads(line) for line in (tmp_path / 'journal.jsonl').read_text().splitlines()]
    assert [r['state'] for r in rows] == ['prepared', 'replaced']
    assert rows[0]['old_sha256'] == sha(original)
    assert rows[1]['sha256'] == sha(path.read_bytes())


@pytest.mark.parametrize('change', ['original', 'replacement'])
def test_changed_input_is_not_replaced(tmp_path, change):
    path = tmp_path / 'primary_depth_0000.npz'
    source_archive(path)
    compressed, receipt = prepare(path)
    if change == 'original':
        path.write_bytes(path.read_bytes() + b'changed after preparation')
    else:
        compressed += b'changed after preparation'
    before = path.read_bytes()
    with (tmp_path / 'journal.jsonl').open('x') as journal:
        with pytest.raises(ValueError, match='changed'):
            replace(path, compressed, receipt, journal)
    assert path.read_bytes() == before
    assert not path.with_name(path.name + '.lzma-pending').exists()
