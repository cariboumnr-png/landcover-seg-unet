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
Domain-tile mapping for categorical rasters to a world-grid layout.

This module provides data structures and utilities for transforming a
categorical raster into a stable, grid-aligned representation of per-
tile domain features. It standardizes how spatial label distributions
are summarized, filtered, and encoded for downstream modeling.

Core functionality includes:
- Alignment of a world grid to raster space using integer pixel offsets
  (assuming shared CRS and resolution).
- Extraction and validation of per-tile label arrays.
- Global label discovery and remapping to a compact index space.
- Tile filtering based on valid pixel ratios.
- Computation of majority class statistics per tile.
- Construction of normalized class-frequency vectors.
- Dimensionality reduction via PCA to produce compact feature vectors.

The resulting `DomainTileMap` provides a reproducible mapping from grid
coordinates to domain descriptors, enabling consistent conditioning,
feature engineering, and cross-dataset alignment.

Persistence is supported via JSON payloads with schema versioning and
integrity checks.
'''

# standard imports
from __future__ import annotations
import collections.abc
import typing

# ---------------------------------Public Type---------------------------------
class DomainPayload(typing.TypedDict):
    '''
    Serializable artifact for `DomainTileMap`.

    This follows the standard artifact contract:

        {
            "schema_id": str,
            "artifact_meta": _DomainMetadata,
            "data": dict[str, DomainTile]
        }

    The DomainTileMap stores per-tile domain features computed from a
    categorical raster aligned to a world grid. The mapping is spatially
    indexed and typically used for downstream conditioning or feature
    extraction.

    Fields:

    - **schema_id**:
        Versioned identifier describing the serialization contract.
    - **artifact_meta**:
        Lightweight derived metadata summarizing dataset-level statistics
        and transformation results (e.g., PCA configuration and class
        frequency summaries).
    - **data**:
        Mapping of string-encoded tile coordinates ('x,y') to `DomainTile`
        feature descriptors.
    '''
    schema_id: str
    artifact_meta: DomainMeta
    data: dict[str, DomainTile]

class DomainMeta(typing.TypedDict):
    '''Lightweight metadata describing a `DomainTileMap` artifact.'''
    world_grid_ids: list[str]
    valid_threshold: float
    target_variance: float
    max_index: int
    major_freq_mean: float
    major_freq_min: float
    pca_axes_n: int
    explained_variance: float

class DomainTile(typing.TypedDict):
    '''
    Typed dictionary representing per-tile domain descriptors.

    Each tile captures summary statistics and optional feature vectors
    derived from its underlying categorical label distribution.

    Fields:

    - **majority**:
        Integer class ID of the dominant class in the tile, or None if
        the tile is invalid or filtered out.

    - **major_freq**:
        Fraction of pixels belonging to the majority class, or None if
        unavailable.

    - **pca_feature**:
        Low-dimensional feature vector derived from PCA on the tile's
        normalized class-frequency distribution, or None if not computed.
    '''
    majority: int | None
    major_freq: float | None
    pca_feature: list[float] | None

# --------------------------------Public  Class--------------------------------
class DomainTileMap(collections.abc.Mapping[tuple[int, int], DomainTile]):
    '''
    Mapping from world-grid tile coordinates to per-tile domain features.

    A `DomainTileMap` represents a spatially indexed collection of
    domain descriptors derived from a categorical raster. Each tile is
    associated with summary statistics and optional feature vectors that
    characterize its class distribution.

    Key characteristics:
        - Stable coordinate system using (x_px, y_px) pixel origins
        - Consistent label remapping to a compact index space
        - Filtering of low-quality tiles based on valid pixel ratios
        - Per-tile majority class statistics
        - PCA-based dimensionality reduction of class distributions

    The mapping behaves like a read-only dictionary from coordinate
    tuples to `DomainTile` entries, while allowing controlled internal
    population.

    **Schema**: `SCHEMA_ID` = `'domain_tile_map_payload/v1'`
    '''

    # current schema
    SCHEMA_ID: str = 'domain_tile_map_payload/v1'

    def __init__(self) -> None:
        '''
        Initialize an empty `DomainTileMap`.

        The instance is created with default metadata and an empty
        internal mapping. Population is expected to occur via external
        construction logic or helper class methods such as
        `from_json_payload()` or `from_dict()`.

        Notes: This constructor does not perform any raster processing.
        '''

        # init attrs
        self.meta: DomainMeta = {
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
        '''Return the maximum label index across all tiles.'''
        return self.meta['max_index']

    @property
    def n_pca_ax(self) -> int:
        '''Return the number of PCA components retained.'''
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
        '''
        Convert the mapping to a JSON-serializable payload.

        This includes both global metadata and per-tile descriptors,
        with tile coordinates encoded as strings.

        Returns:
            A `DomainPayload` dictionary suitable for JSON serialization.

        Raises:
            AssertionError:
                If required metadata fields are not properly populated.
        '''

        # sanity checks and return payload
        assert self.meta['major_freq_mean'] > 0.0
        assert self.meta['major_freq_min'] < 1.0
        assert self.meta['pca_axes_n'] > 0
        assert self.meta['explained_variance'] > 0.0
        return {
            'schema_id': self.SCHEMA_ID,
            'artifact_meta': self.meta,
            'data': {f'{k[0]},{k[1]}': v for k, v in self._data.items()}
        }

    @classmethod
    def from_json_payload(cls, payload: DomainPayload) -> DomainTileMap:
        '''
        Reconstruct a `DomainTileMap` from a JSON payload.

        Args:
            payload:
                Dictionary containing serialized metadata and tile data.

        Returns:
            A populated `DomainTileMap` instance.

        Notes: Tile coordinate keys are converted from "x,y" strings back
        into (x, y) integer tuples.
        '''

        def _xy_tuple(inputs: str) -> tuple[int, int]: # e.g., '1,2' -> (1, 2)
            output = tuple(int(x.strip()) for x in inputs.split(','))
            assert len(output) == 2
            return output

        # create empty DomainTile instance
        obj = cls.__new__(cls) # skip __init__()
        # populate attributes from payload as needed
        obj.meta = payload['artifact_meta']
        obj._data = {_xy_tuple(k): v for k, v in payload['data'].items()}
        # return class object
        return obj

    @classmethod
    def from_dict(cls, tiles: dict[tuple[int, int], DomainTile]) -> DomainTileMap:
        '''
        Construct a `DomainTileMap` from an in-memory dictionary.

        Args:
            tiles:
                Dictionary mapping (x, y) coordinate tuples to
                `DomainTile` entries.

        Returns:
            A `DomainTileMap` instance containing the provided data.

        Notes: Metadata is initialized with default values and should be
        updated separately if needed.
        '''

        # create empty DomainTile instance
        self = cls() # don't skip __init__() - need an empty meta dict
        # populate self._data
        self._data = tiles
        # return class object
        return self
