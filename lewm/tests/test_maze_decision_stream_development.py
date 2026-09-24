import gzip
import json
import pytest
from scripts.maze_decision_stream_development import writer, read_rows, NAME


def test_lossless_ordered_stream_and_exclusive_output(tmp_path):
    rows = [dict(tick=i, decision=dict(prediction=[[.1, 0., -1.]], terminal=None), text='actual\nnewlines') for i in range(3)]
    with writer(tmp_path) as append:
        for row in rows: append(row)
    assert list(read_rows(tmp_path)) == rows
    before = (tmp_path/NAME).read_bytes()
    with pytest.raises(FileExistsError):
        with writer(tmp_path): pass
    assert (tmp_path/NAME).read_bytes() == before


@pytest.mark.parametrize('bad', [dict(tick=False), dict(tick=1), dict(tick=0, value=float('nan'))])
def test_invalid_receipt_cannot_be_written(tmp_path, bad):
    with writer(tmp_path) as append:
        with pytest.raises(ValueError): append(bad)
    assert list(read_rows(tmp_path)) == []


def test_reader_rejects_partial_line_and_missing_frame(tmp_path):
    with gzip.open(tmp_path/NAME, 'wb') as f: f.write(json.dumps(dict(tick=0)).encode())
    with pytest.raises(ValueError): list(read_rows(tmp_path))
    (tmp_path/NAME).unlink()  # Own synthetic fixture only.
    with gzip.open(tmp_path/NAME, 'wb') as f: f.write(b'{"tick":1}\n')
    with pytest.raises(ValueError): list(read_rows(tmp_path))
