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
import typing

# ---------------------------------Public Type---------------------------------
class DomainTile(typing.TypedDict):
    '''Per-tile domain descriptors stored in a `DomainTileMap`.'''
    majority: int | None
    major_freq: float | None
    pca_feature: list[float] | None

class DomainMetadata(typing.TypedDict):
    '''doc'''
    schema_id: str
    world_grid_ids: list[str]
    valid_threshold: float
    target_variance: float
    max_index: int
    major_freq_mean: float
    major_freq_min: float
    pca_axes_n: int
    explained_variance: float

class DomainPayload(typing.TypedDict):
    '''Serializable payload for `DomainTileMap` persistence.'''
    meta: DomainMetadata
    tiles_dict: dict[str, DomainTile]

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

    def __init__(self) -> None:
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
        self.meta: DomainMetadata = {
            'schema_id': self.SCHEMA_ID,
            'world_grid_ids': [],
            'valid_threshold': 1.0,
            'target_variance': 1.0,
            'max_index': -1,
            'major_freq_mean': 0.0,
            'major_freq_min': 1.0,
            'pca_axes_n': 0,
            'explained_variance': 0.0,
        }
        self._data: dict[tuple[int, int], DomainTile] = {}

    @property
    def max_id(self) -> int:
        '''Gloabl max index.'''
        return self.meta['max_index']

    @property
    def n_pca_ax(self) -> int:
        '''Number of PCA axes that achieves target variance.'''
        return self.meta['pca_axes_n']

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
        return len(self._data)

    # ----- representation
    def __str__(self) -> str:
        return '\n'.join([
            'Domain details:',
            f'Tile filtering threshold:  {self.meta["valid_threshold"]:.2f}',
            f'Number of domain tiles:    {len(self)}',
            f'Mean freq. of major class: {self.meta["major_freq_mean"]:.2f}',
            f'Min freq. of major class   {self.meta["major_freq_min"]:.2f}',
            f'Target PCA variance:       {self.meta["target_variance"]}',
            f'PCA axes count:            {self.meta["pca_axes_n"]}',
            f'PCA variance explained:    {self.meta["explained_variance"]:.2f}'
        ])

    # ----- public method
    def to_json_payload(self) -> DomainPayload:
        '''Generate class payload for JSON serialization.'''

        # sanity checks and return payload
        assert self.meta['major_freq_mean'] > 0.0
        assert self.meta['major_freq_min'] < 1.0
        assert self.meta['pca_axes_n'] > 0
        assert self.meta['explained_variance'] > 0.0
        return {
            'meta': self.meta,
            'tiles_dict': {f'{k[0]},{k[1]}': v for k, v in self._data.items()}
        }

    @classmethod
    def from_json_payload(cls, payload: DomainPayload) -> DomainTileMap:
        '''Instantiate a class object with input JSON payload.'''

        def _xy_tuple(inputs: str) -> tuple[int, int]: # e.g., '1,2' -> (1, 2)
            output = tuple(int(x.strip()) for x in inputs.split(','))
            assert len(output) == 2
            return output

        # create empty DomainTile instance
        obj = cls.__new__(cls) # skip __init__()
        # populate attributes from payload as needed
        obj.meta = payload['meta']
        obj._data = {_xy_tuple(k): v for k, v in payload['tiles_dict'].items()}
        # return class object
        return obj

    @classmethod
    def from_dict(cls, tiles: dict[tuple[int, int], DomainTile]) -> DomainTileMap:
        '''Instantiate a class object with an in-memory tile dict.'''

        # create empty DomainTile instance
        self = cls() # don't skip __init__() - need an empty meta dict
        # populate self._data
        self._data = tiles
        # return class object
        return self
