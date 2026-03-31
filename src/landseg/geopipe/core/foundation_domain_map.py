# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

'''
Domain-tile mapping utilities for categorical rasters aligned to a
world-grid layout.

This module defines the `DomainTileMap` class, its supporting data
structures, and the full pipeline for converting a categorical raster
into stable, per-tile domain features:

    - Align a deep-copied world grid to the raster by computing an
      integer pixel offset. The raster must share the grid's CRS and
      pixel size.
    - Read all grid windows (in parallel) and extract per-tile integer
      label arrays, enforcing integer dtype and a well-defined nodata
      value.
    - Discover all unique valid labels globally; validate against the
      configured index_base, then remap labels to a compact [0..K-1]
      space for consistency across tiles.
    - Filter tiles based on the fraction of valid pixels. For each valid
      tile, compute majority-class id and frequency and update global
      summary statistics in the `DomainContext`.
    - Construct normalized class-frequency vectors for valid tiles and
      apply PCA (via economical SVD) to produce low-dimensional float32
      feature vectors that meet a target cumulative explained variance.
    - Assemble a `DomainTileMap` containing majority statistics, PCA
      features, and context metadata.

Persistence is handled through JSON payloads and JSON metadata files.
A schema identifier ('domain_tile_map_payload/v1') and SHA256 hash ensure
compatibility and integrity upon reload.

Tile coordinates follow grid conventions: (x_px, y_px) pixel-origin
coordinates relative to the grid origin. All mapping operations preserve
this stable coordinate space, which enables reproducible conditioning and
cross-dataset alignment.
'''

# standard imports
from __future__ import annotations
import collections.abc
import dataclasses
import typing
# third-party imports
import numpy
# local imports
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.foundation.domain_maps as domain_maps
import landseg.utils as utils

# ---------------------------------Public Type---------------------------------
class DomainTile(typing.TypedDict):
    '''Per-tile domain descriptors stored in a `DomainTileMap`.'''
    majority: int | None
    major_freq: float | None
    pca_feature: list[float] | None

class DomainPayload(typing.TypedDict):
    '''Serializable payload for `DomainTileMap` persistence.'''
    blk_size: list[int]
    blk_overlap: list[list[int]]
    idx_range: list[int]
    context: dict[str, typing.Any]
    valid_idx: list[list[int]]
    tiles: dict[str, typing.Any]

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _DomainContext:
    '''Run-time context and summary statistics for a `DomainTileMap`.'''
    valid_threshold: float
    target_variance: float
    major_freq_mean: float
    major_freq_min: float
    pca_axes_n: int
    explained_variance: float

# --------------------------------Public  Class--------------------------------
class DomainTileMap(collections.abc.Mapping[tuple[int, int], DomainTile]):
    '''
    Mapping from world-grid tile coordinates to per-tile domain features.

    A `DomainTileMap` materializes domain knowledge over a fixed world
    grid. It aligns the grid to a categorical raster, remaps raw labels
    to a compact [0..K-1] space, filters tiles by valid-pixel fraction,
    computes majority statistics, derives normalized class-frequency
    vectors, and projects them via PCA to reach a target explained
    variance.

    Key space:
    - (x_px, y_px) pixel-origin coordinates relative to the grid origin.

    Schema:
        SCHEMA_ID = 'domain_tile_map_payload/v2'
    '''

    # current schema
    SCHEMA_ID: str = 'domain_tile_map_payload/v2'

    def __init__(
        self,
        valid_threshold: float,
        target_variance: float,
        logger: utils.Logger
    ) -> None:
        '''
        Build a `DomainTileMap` from categorical raster and world grid.

        Args:
            domain_fpath: Path to a single-band integer raster of class
                labels. Its CRS and pixel size must match those of
                `world_grid`.
            world_grid: A grid.GridLayout instance that defines tile
                windows. The grid is deep-copied and aligned to
                `domain_fpath` via an integer pixel offset before reads.
            context: `DomainContext` carrying thresholds and PCA target
                variance. This object is updated in-place with statistics
                collected during build.
            logger: Logger for progress and summary messages.

        Notes:
        The constructor reads all tile windows, performs label discovery
        and remapping, filters tiles by 'valid_threshold', computes
        majority statistics, runs PCA on normalized frequency vectors,
        and stores results in the internal mapping.
        '''

        # init attrs
        self._ctx = _DomainContext(
            valid_threshold=valid_threshold,
            target_variance=target_variance,
            major_freq_mean = 0.0,
            major_freq_min = 1.0,
            pca_axes_n = 0,
            explained_variance = 0.0
        )
        self.index_range: tuple[int, int]
        self.blk_size: tuple[int, int]
        self.blk_overlaps: list[tuple[int, int]]
        self._valid: list[tuple[int, int]] = []
        self._data: dict[tuple[int, int], DomainTile] = {}
        self.logger = logger

    @property
    def max_id(self) -> int:
        '''Gloabl max index.'''
        return self.index_range[1]

    @property
    def n_pca_ax(self) -> int:
        '''Number of PCA axes that achieves target variance.'''
        return self._ctx.pca_axes_n

    # ----- container protocol
    def __getitem__(self, idx: tuple[int, int]) -> DomainTile:
        # fail fast on idx type check
        if (not isinstance(idx, tuple) or len(idx) != 2
            or not all(isinstance(v, int) for v in idx)):
            raise TypeError('Index must be (x, y) in pixels as integers.')
        return self._data[idx]

    def __iter__(self) -> collections.abc.Iterator[tuple[int, int]]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._valid)

    # ----- representation
    def __str__(self) -> str:
        return '\n'.join([
            'Domain details:',
            f'Tile filtering threshold: {self._ctx.valid_threshold:.2f}',
            f'Number of valid domain tiles: {len(self)}',
            f'Mean frequency of major class: {self._ctx.major_freq_mean:.2f}',
            f'Min frequency of major class {self._ctx.major_freq_min:.2f}',
            f'Target PCA variance: {self._ctx.target_variance}',
            f'PCA axes count: {self._ctx.pca_axes_n}',
            f'PCA variance explained: {self._ctx.explained_variance:.2f}',
        ])

    # ----- public method
    def to_payload(self) -> DomainPayload:
        '''Generate class payload for json dump.'''

        # sanity checks and return payload
        assert self._ctx.major_freq_mean > 0.0
        assert self._ctx.major_freq_min < 1.0
        assert self._ctx.pca_axes_n > 0
        assert self._ctx.explained_variance > 0.0
        return {
            'blk_size': list(self.blk_size),
            'blk_overlap': [list(o) for o in self.blk_overlaps],
            'idx_range': list(self.index_range),
            'context': dataclasses.asdict(self._ctx),
            'valid_idx': [list(x) for x in self._valid],
            'tiles': {f'{k[0]},{k[1]}': v for k, v in self._data.items()}
        }

    @classmethod
    def from_payload(
        cls, payload: DomainPayload,
        logger: utils.Logger
    ) -> DomainTileMap:
        '''Instantiate the class with input payload.'''

        def _to_xy_tuple(inputs: list[int] | str) -> tuple[int, int]:
            # [1, 2] | '1,2' -> (1, 2)
            if isinstance(inputs, str):
                output = tuple(int(x.strip()) for x in inputs.split(','))
                assert len(output) == 2
                return output
            assert len(inputs) == 2
            assert all(isinstance(x, int) for x in inputs)
            return inputs[0], inputs[1]

        # create empty GridLayout instance
        obj = cls.__new__(cls) # skip __init__()
        # populate attributes from payload as needed
        obj.blk_size = _to_xy_tuple(payload['blk_size'])
        obj.blk_overlaps = [_to_xy_tuple(o) for o in payload['blk_overlap']]
        obj.index_range = _to_xy_tuple(payload['idx_range'])
        obj._ctx = _DomainContext(**payload['context'])
        obj._valid = [_to_xy_tuple(x) for x in payload['valid_idx']]
        obj._data = {_to_xy_tuple(k): v for k, v in payload['tiles'].items()}
        obj.logger = logger
        # return class object
        return obj

    def build(self,
        block_size: tuple[int, int],
        block_overlap: tuple[int, int],
        index_range: tuple[int, int],
        raster_tiles: alias.RasterTileDict
    ) -> None:
        '''
        Compute per-tile statistics and PCA features, and fill the map.

        Steps:
        1) Initialize a DomainTile template for each tile.
        2) Select valid tiles using 'valid_threshold'.
        3) For valid tiles, compute 'majority' and 'major_freq' and
            update 'major_freq_min' and 'major_freq_mean' in the context.
        4) Build normalized frequency vectors for valid tiles, run
            PCA to reach 'target_variance', and store 'pca_feature' per
            valid tile.
        '''

        # early exit if all input tiles are already in map
        if set(raster_tiles.keys()).issubset(set(self._data.keys())):
            self.logger.log('INFO', 'Input tiles already mapped, exit build')
            return

        # args directly to attrs - if no attrs present this is a new map
        if not hasattr(self, 'index_range'):
            self.index_range = index_range
        if not hasattr(self, 'blk_size'):
            self.blk_size = block_size
        if not hasattr(self, 'blk_overlaps'):
            self.blk_overlaps = [block_overlap]
        else:
            self.blk_overlaps.append(block_overlap)
        print(self.blk_overlaps)

        # first iteration to get indices of valid tiles and index range
        self.logger.log('INFO', 'Filter input raster tiles')
        for c, tile in raster_tiles.items():
            if c in self._data: # skip keys that already in data
                continue
            # add all to self._data
            self._data[c] = {
                'majority': None,
                'major_freq': None,
                'pca_feature': None,
            }
            if _is_valid(tile, self._ctx.valid_threshold):
                # add to valid coordinates list
                self._valid.append(c)

        # get majority index for valid tiles - calc here
        self.logger.log('INFO', 'Calculate majority class from new tiles')
        for c in set(raster_tiles.keys()):
            if c in self._data: # skip keys that already in data
                continue
            tile = raster_tiles[c]
            values, counts = numpy.unique(tile, return_counts=True)
            # update domain tile dict
            major = values[numpy.argmax(counts)].item() # serializable
            freq = counts[numpy.argmax(counts)] / sum(counts)
            self._data[c].update({'majority': major, 'major_freq': freq})
            # update major_freq stats
            self._ctx.major_freq_min = min(self._ctx.major_freq_min, freq)
            self._ctx.major_freq_mean += freq
        self._ctx.major_freq_mean /= len(self._valid)

        # get pca transform for valid tiles - calc delegated to transform.py
        self.logger.log('INFO', 'PCA transforming all valid tiles')
        freqs: dict[tuple[int, int], numpy.ndarray] = {}
        for c in self._valid: # calculate on all valid tiles
            tile = raster_tiles[c]
            freqs[c] = _norm_freq(tile, (self.index_range))
        # get full pca
        z, evr, k = domain_maps.pca_transform(freqs, self._ctx.target_variance)
        self._ctx.explained_variance = evr
        self._ctx.pca_axes_n = k
        # assign to each valid tile
        for c in self._valid:
            tile = raster_tiles[c]
            self._data[c].update({'pca_feature': [float(x) for x in z[c]]})

# ------------------------------private  function------------------------------
def _norm_freq(
    arr: numpy.ndarray,
    index_range: tuple[int, int],
) -> numpy.ndarray:
    '''Get a normalized class frequency vector from the array.'''

    # get frequencies of valid elements
    valid = arr[arr != -1]
    values, counts = numpy.unique(valid, return_counts=True)
    frequencies = counts / counts.sum()
    # map class value to frequency
    freq_map = dict(zip(values, frequencies))
    i, j = index_range
    return numpy.array([freq_map.get(idx, 0.0) for idx in range(i, j + 1)])

def _is_valid(
    arr: numpy.ndarray,
    threshold: float
) -> bool:
    '''Return True if fraction of valid pixels >= threshold.'''

    valid = arr != -1
    if valid.size == 0:
        return False
    return float(valid.mean()) >= float(threshold)
