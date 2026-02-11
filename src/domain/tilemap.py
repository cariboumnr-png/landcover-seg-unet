'''
Domain-tile mapping utilities for categorical rasters aligned to a
world-grid layout.

This module defines the `DomainTileMap` class, its supporting data
structures, and the full pipeline for converting a categorical raster
into stable, per-tile domain features:

    - Align a deep-copied world grid to the raster by computing an
      integer pixel offset. The raster must share the grid’s CRS and
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
import copy
import dataclasses
import typing
# third-party imports
import numpy
import rasterio
# local imports
import alias
import domain
import grid
import utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class DomainContext:
    '''Run-time context and summary statistics for a `DomainTileMap`.'''
    index_base: int
    valid_threshold: float
    target_variance: float
    major_freq_mean: float = 0.0
    major_freq_min: float = 1.0
    pca_axes_n: int = 0
    explained_variance: float = 0.0

# ---------------------------------Public Type---------------------------------
class DomainTile(typing.TypedDict):
    '''Per-tile domain descriptors stored in a `DomainTileMap`.'''
    majority: int | None
    major_freq: float | None
    pca_feature: list[float] | None

class DomainPayload(typing.TypedDict):
    '''Serializable payload for `DomainTileMap` persistence.'''
    context: dict[str, typing.Any]
    valid_idx: list[str]
    tiles: dict[str, typing.Any]

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
        SCHEMA_ID = 'domain_tile_map_payload/v1'
    '''

    # current schema
    SCHEMA_ID: str = 'domain_tile_map_payload/v1'
    # tile template
    TILE_TEMP: DomainTile = {
        'majority': None,
        'major_freq': None,
        'pca_feature': None,
    }

    def __init__(
        self,
        domain_fpath: str,
        world_grid: grid.GridLayout,
        context: DomainContext,
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

        # parse argument
        self._src = domain_fpath
        self._grid = copy.deepcopy(world_grid) # make a deepcopy
        self._ctx = context
        # init attrs
        self._range: tuple[int, int] = (0, 0)
        self._valid: list[tuple[int, int]] = []
        self._data: dict[tuple[int, int], DomainTile] = {}
        self.logger = logger
        # finish initialization
        all_tiles = self._map_domain_to_grid()
        self._populate(all_tiles)

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
            'context': dataclasses.asdict(self._ctx),
            'valid_idx': [f'{x[0]}, {x[1]}' for x in self._valid],
            'tiles': {f'{k[0]}, {k[1]}': v for k, v in self._data.items()}
        }

    @classmethod
    def from_payload(cls, payload: DomainPayload) -> DomainTileMap:
        '''Instantiate the class with input payload.'''

        # create empty GridLayout instance
        obj = cls.__new__(cls)
        # populate attributes from payload as needed
        obj._ctx = DomainContext(**payload['context'])
        obj._valid = [_parse_coords(x) for x in payload['valid_idx']]
        obj._data = {_parse_coords(k): v for k, v in payload['tiles'].items()}
        # return class object
        return obj

    # ----- private method
    def _map_domain_to_grid(self) -> list[alias.RasterTile]:
        '''
        Read the domain raster over all grid windows and remap labels.

        Behavior:
        - Align the deep-copied world grid to the raster by computing an
        integer-pixel offset.
        - Discover all unique labels across tiles (excluding nodata);
        assert the set is non-empty and its minimum equals 'index_base'.
        - Build a sorted 'remap' array and convert valid labels to
        [0..K-1]; assign -1 to nodata.
        '''

        # open domain raster
        with rasterio.open(self._src) as src:
            # offset the world grid to align with input raster
            self._grid.set_offset_from(src)
            # infer dtype and nodata
            dtype = numpy.dtype(src.dtypes[0])
            assert numpy.issubdtype(dtype, numpy.integer) # sanity check: int only
            nodata = src.nodata
            if nodata is None:
                nodata = -1 # default value
            assert abs(nodata - round(nodata)) < 1e-9 # sanity check: nodata is int

        # read through all windows via multiprocessing
        jobs = [(_read_window, (k, v, self._src), {}) for k, v in self._grid.items()]
        all_tiles: list[alias.RasterTile] = utils.ParallelExecutor().run(jobs)

        # first iteration on all tiles
        # gather unique values (exclude nodata)
        unique_values = set()
        for (_, arr) in all_tiles:
            unique_values.update(numpy.unique(arr))
        if nodata in unique_values:
            unique_values.remove(nodata) # remove nodata if present
        # safety check against empty input raster
        if not unique_values:
            raise ValueError("No valid data values found.")
        # make sure minimal aligns with index base
        if min(unique_values) != self._ctx.index_base:
            raise ValueError(
                f'Min value {min(unique_values)} != base {self._ctx.index_base}'
            )

        # global mapping: raw i..K  ->  0..K-1 ----
        remap = numpy.array(sorted(unique_values), dtype=numpy.int64)
        self._range = (0, remap.size - 1) # get range

        # second iteration on all tiles
        # Map each block: valid raw -> index in [0..K-1], nodata -> -1
        for (_, arr) in all_tiles:
            mask_valid = arr != nodata
            # skip if a grid tile does not contain any data
            if not numpy.any(mask_valid):
                arr[...] = -1 # ensure to -1
                continue
            # get indices of insertion
            idx = numpy.searchsorted(remap, arr[mask_valid])
            # safety checks
            if not numpy.all(remap[idx] == arr[mask_valid]):
                raise ValueError('Encountered value not present in remap domain')
            if numpy.any(idx >= remap.size):
                raise ValueError('Encountered value outside remap domain')
            # assgin values to array
            arr[mask_valid] = idx
            arr[~mask_valid] = -1

        return all_tiles

    def _populate(self, all_tiles: list[alias.RasterTile]) -> None:
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

        # give all tiles a copy of the template
        for (coord, _) in all_tiles:
            self._data[coord] = copy.deepcopy(self.TILE_TEMP)

        # filter once to get indices of valid tiles
        valid_indices: list[int] = []
        for i, (coords, arr) in enumerate(all_tiles):
            if _is_valid(arr, self._ctx.valid_threshold):
                valid_indices.append(i)
                self._valid.append(coords)

        # get majority index for valid tiles - calc here
        for i in valid_indices:
            coords, arr = all_tiles[i]
            values, counts = numpy.unique(arr, return_counts=True)
            # update domain tile dict
            major = values[numpy.argmax(counts)].item() # serializable
            freq = counts[numpy.argmax(counts)] / sum(counts)
            self._data[coords].update({'majority': major, 'major_freq': freq})
            # update major_freq stats
            self._ctx.major_freq_min = min(self._ctx.major_freq_min, freq)
            self._ctx.major_freq_mean += freq
        self._ctx.major_freq_mean /= len(valid_indices)

        # get pca transform for valid tiles - calc delegated to transform.py
        # get index frequency for valid tiles
        freqs: dict[tuple[int, int], numpy.ndarray] = {}
        for i in valid_indices:
            coords, arr = all_tiles[i]
            freqs[coords] = _norm_freq(arr, self._range)
        # get full pca
        z, evr, k = domain.pca_transform(freqs, self._ctx.target_variance)
        self._ctx.explained_variance = evr
        self._ctx.pca_axes_n = k
        # assign to each valid tile
        for i in valid_indices:
            coords, arr = all_tiles[i]
            self._data[coords].update(
                {'pca_feature': [float(x) for x in z[coords]]}
            )

# ------------------------------private  function------------------------------
def _read_window(
        raster_window_id: tuple[int, int],
        raster_window: alias.RasterWindow,
        raster_fpath: str
    ) -> alias.RasterTile:
    '''Read domain raster via a window and return the first channel.'''

    # read raster at window and return
    with rasterio.open(raster_fpath, 'r') as src:
        arr = src.read(1, window=raster_window) # [1, H, W] for 2D rasters
        arr = arr.astype(numpy.int64, copy=False) # ensure negatives support
        return raster_window_id, arr

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
    # 1-based, inclusive
    i, j = index_range
    # return
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

def _parse_coords(coord_str: str) -> tuple[int, int]:
    '''Parse coords in str as `"x,y"` back to `tuple[int, int]`.'''

    x, y = coord_str.split(',')
    return int(x), int(y)
