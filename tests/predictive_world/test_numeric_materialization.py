"""Batch extraction must preserve legacy numeric labels, masks and row order."""
from collections import OrderedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from psi.data.predictive_world.dataset import PredictiveWorldRowReader, _read_vectors, _vector


def legacy(table, positions, columns):
    return np.stack([
        np.concatenate([_vector(table[column][index]) for column in columns])
        for index in positions
    ]).astype(np.float32, copy=False)


@pytest.mark.parametrize('kind', ['fixed', 'list', 'large', 'scalar', 'nested'])
def test_arrow_gather_exactly_matches_scalar_path_with_chunks_and_slices(kind):
    data = [[0., -0., 2.], [3., np.nan, 5.], [np.inf, -np.inf, 8.], [9., 10., 11.]]
    if kind == 'fixed':
        typ = pa.list_(pa.float32(), 3)
    elif kind == 'list':
        typ = pa.list_(pa.float64())
    elif kind == 'large':
        typ = pa.large_list(pa.float32())
    elif kind == 'scalar':
        data = [0., None, np.inf, -0.]
        typ = pa.float32()
    else:
        data = [[[v] for v in row] for row in data]
        typ = pa.list_(pa.list_(pa.float64()))
    chunks = pa.chunked_array([pa.array(data[:2], type=typ), pa.array(data[2:], type=typ)])
    table = pa.table({'state': chunks, 'action': [0, 1, 2, 3], 'unused': ['a'] * 4}).slice(1)
    positions = [2, 0, 2, 1, -1]
    columns = ['action', 'state']
    actual = _read_vectors(table, positions, columns)
    expected = legacy(table, positions, columns)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_parent_null_lists_and_selected_ragged_rows_keep_scalar_behavior():
    table = pa.table({'v': pa.array([[1., 2.], [3.], None], type=pa.list_(pa.float32()))})
    for positions in [[1, 2, 1], [0, 0]]:
        np.testing.assert_array_equal(_read_vectors(table, positions, ['v']), legacy(table, positions, ['v']))
    with pytest.raises(ValueError):
        _read_vectors(table, [0, 1], ['v'])


def test_empty_and_null_element_lists_preserve_width_and_nonfinite_values():
    table = pa.table({'empty': pa.array([[], []], type=pa.list_(pa.float32())),
                      'values': pa.array([[None, 2.], [3., None]], type=pa.list_(pa.float32()))})
    for columns in [['empty'], ['empty', 'values']]:
        np.testing.assert_array_equal(_read_vectors(table, [1, 0, 1], columns), legacy(table, [1, 0, 1], columns))


@pytest.mark.parametrize('positions', [[3], [-4], []])
def test_bad_numeric_indices_fail_instead_of_reading_another_row(positions):
    with pytest.raises((IndexError, ValueError)):
        _read_vectors(pa.table({'v': [0, 1, 2]}), positions, ['v'])


def test_six_shards_stay_cached_without_changing_selected_labels(tmp_path, monkeypatch):
    paths = []
    for shard in range(6):
        path = tmp_path / f'{shard}.parquet'
        pq.write_table(pa.table({'v': [[float(shard), 2.], [3., float(shard)]]}), path)
        paths.append(path)
    reads = []
    original = pq.read_table

    def read(path, **kwargs):
        reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(pq, 'read_table', read)
    reader = PredictiveWorldRowReader.__new__(PredictiveWorldRowReader)
    reader._tables = OrderedDict()
    reader._parquet_cache_capacity = 6
    for _ in range(3):
        for path in paths:
            table = reader._table(str(path), ('v',))
            np.testing.assert_array_equal(_read_vectors(table, [1, 0], ['v']), legacy(table, [1, 0], ['v']))
    assert len(reads) == 6
    reader._table(str(paths[0]), ('v',))
    reader._parquet_cache_capacity = 2
    # Reducing capacity evicts old entries on the next miss; it never changes values.
    extra = tmp_path / 'extra.parquet'
    pq.write_table(pa.table({'v': [[7., 8.]]}), extra)
    reader._table(str(extra), ('v',))
    assert len(reader._tables) == 2
