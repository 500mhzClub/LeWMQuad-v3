"""Use row-tiled dense arithmetic only where the existing router selects dense."""
from types import FunctionType
from lewm import density_routed_floor_cell_index_development as original
from lewm.tiled_dense_floor_cell_index_development import observed_floor_cell_index as tiled_dense_index

_function=original.observed_floor_cell_index
observed_floor_cell_index=FunctionType(_function.__code__,
    _function.__globals__|dict(dense_index=tiled_dense_index),
    _function.__name__,_function.__defaults__,_function.__closure__)
observed_floor_cell_index.__kwdefaults__=_function.__kwdefaults__
