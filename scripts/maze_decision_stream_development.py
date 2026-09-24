"""Lossless bounded exclusive gzip JSON-lines for large online decision receipts."""
import gzip
import json
from contextlib import contextmanager
from lewm.causal_rgb_dataset_development import _leaf
from lewm.novel_maze_round_trip_contract_development import MAX_OBSERVATIONS

NAME = 'context_decisions.jsonl.gz'
MAX_ROW_BYTES = 32*1024**2


@contextmanager
def writer(directory):
    with (directory/NAME).open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            count = 0
            def append(row):
                nonlocal count
                if count >= MAX_OBSERVATIONS or type(row['tick']) is not int or row['tick'] != count:
                    raise ValueError('bounded consecutive decision rows required')
                payload = (json.dumps(row, separators=(',', ':'), allow_nan=False)+'\n').encode()
                if len(payload) > MAX_ROW_BYTES: raise ValueError('bounded decision row exceeded')
                compressed.write(payload); compressed.flush(); raw.flush(); count += 1
            yield append


def read_rows(directory):
    count = 0
    with gzip.open(_leaf(directory, NAME), 'rb') as stream:
        while True:
            line = stream.readline(MAX_ROW_BYTES+1)
            if not line: break
            if len(line) > MAX_ROW_BYTES or not line.endswith(b'\n') or count >= MAX_OBSERVATIONS:
                raise ValueError('bounded complete decision line required')
            row = json.loads(line)
            if type(row.get('tick')) is not int or row['tick'] != count:
                raise ValueError('consecutive recorded decision identity required')
            yield row
            count += 1
