"""Map domain derived from the generator envelope, never sampled episodes."""
import math

# Same family bounds as independent_round_trip_layouts_development.pack.
GENERATOR_BOUNDS_XY_M = ((-2.1, -2.1), (3.4, 3.4))
GENERATOR_DIAGONAL_M = math.hypot(*(hi-lo for lo, hi in zip(*GENERATOR_BOUNDS_XY_M)))
# Every origin inside that envelope and every heading: distance <= diagonal.
# Retain the deployed 0.1-m coordinate guard within the storage domain.
MAP_HALF_WIDTH_M = float(math.ceil(GENERATOR_DIAGONAL_M + .1))
POINT_BOUND_M = MAP_HALF_WIDTH_M - .1
COARSE_CELL_M = .05
FINE_CELL_M = .01
COARSE_HALF_CELLS = round(MAP_HALF_WIDTH_M / COARSE_CELL_M)
FINE_HALF_CELLS = round(MAP_HALF_WIDTH_M / FINE_CELL_M)
COARSE_CELL_COUNT = (2*COARSE_HALF_CELLS)**2
