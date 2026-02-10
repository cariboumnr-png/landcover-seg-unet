'''
Resolve a domain raster into per-tile stats and conditioning features.

Overview
--------
This module parses a domain raster into a set of spatial tiles defined
by a `GridLayout`. For each tile, raster data within the tile's bounds
is retrieved, remapped to a 0..K-1 class index space, and processed to
compute:

- majority class and its relative frequency
- PCA-based conditioning vectors at explained-variance thresholds

Tiles correspond to fixed spatial units defined by the grid. Internally,
each tile is read using a raster window, but operationally all functions
treat tiles as logical grid units rather than raster I/O constructs.

Processing steps
----------------
1. Parse the raster into tiles using the GridLayout.
2. Remap raw raster values to 0..K-1 based on global unique classes.
3. For each valid tile (sufficient fraction of non-nodata pixels):
   - compute majority label and majority frequency
   - compute a normalized class-frequency vector
4. Run PCA on all valid tiles once, then project tiles to:
   - PCA90  (>= 90% cumulative variance)
   - PCA95  (>= 95% cumulative variance)
   - PCA99  (>= 99% cumulative variance)
5. Assemble all outputs into a list of tile-indexed dictionaries.

Notes
-----
- Invalid tiles (insufficient valid pixels) retain None/empty statistics.
- PCA computation is delegated to the domain PCA utility module.
- Remapping ensures all tiles share the same fixed class index domain.
'''

#standard imports
import copy
# third-party imports
import numpy
import rasterio
# local imports
import alias
import domain
import grid
import utils

# -------------------------------Public Function-------------------------------
def map_domain_to_grid(
    domain_fpath: str,
    world_grid: grid.GridLayout,
    *,
    index_base: int = 1,
    valid_pixel_threshold: float = 0.7
) -> list[dict]:
    '''
    Parse a domain raster into tiles and compute majority and PCA-based
    conditioning features.

    Args:
        domain_fpath:
            Path to the discrete domain raster.
        world_grid:
            GridLayout defining tile coordinate keys and their spatial extents.
        index_base:
            Expected lowest raw class value in the raster (default 1).
        valid_pixel_threshold:
            Minimum fraction of non-nodata pixels required for a tile to be
            considered valid and included in PCA.

    Returns:
        A dict mapping tile coordinates (e.g., (row, col)) to a result dict
        containing fields:
            - 'coords'
            - 'majority'
            - 'major_freq'
            - 'pca90'
            - 'pca95'
            - 'pca99'
    '''

    # parse domain
    parsed, max_idx = _parse_domain(domain_fpath, world_grid, index_base)

    # init an output dict: give all tiles a copy of dict
    output: dict[tuple[int, int], dict] = {}
    for (coords, _) in parsed:
        output[coords] = copy.deepcopy({
            'coords': coords,
            'majority': None,
            'major_freq': None,
            'pca90': None,
            'pca95': None,
            'pca99': None
        })

    # get majority index for valid tiles - calc here
    for (coords, arr) in parsed:
        # pass if invalid
        if not is_valid(arr, valid_pixel_threshold):
            continue
        values, counts = numpy.unique(arr, return_counts=True)
        # update domain tile dict
        output[coords].update({
            'majority': values[numpy.argmax(counts)].item(), # serializable
            'major_freq': counts[numpy.argmax(counts)] / sum(counts)
        })

    # get pca transform for valid tiles - calc delegated to transform.py
    # get index frequency for valid tiles
    freqs: dict[tuple[int, int], numpy.ndarray] = {}
    for (coords, arr) in parsed:
        # pass if invalid
        if not is_valid(arr, valid_pixel_threshold):
            continue
        freqs[coords] = _norm_freq(arr, (0, max_idx - 1))
    # get pca
    pca90, _ = domain.pca_transform(freqs, 0.9)
    pca95, _ = domain.pca_transform(freqs, 0.95)
    pca99, _ = domain.pca_transform(freqs, 0.99)
    # loop again and update
    for (coords, arr) in parsed:
        # pass if invalid
        if not is_valid(arr, valid_pixel_threshold):
            continue
        output[coords].update({
            'pca90': [float(x) for x in pca90[coords]],
            'pca95': [float(x) for x in pca95[coords]],
            'pca99': [float(x) for x in pca99[coords]]
        })

    # return
    return list(output.values())

# ------------------------------private  function------------------------------
def _parse_domain(
    domain_fpath: str,
    world_grid: grid.GridLayout,
    index_base: int = 1,
) -> tuple[list[alias.RasterTile], int]:
    '''doc'''

    # make a deepcopy of world grid
    _grid = copy.deepcopy(world_grid)

    # open domain raster
    with rasterio.open(domain_fpath) as src:
        # offset the world grid to align with input raster
        _grid.set_offset_from(src)
        # infer dtype and nodata
        dtype = numpy.dtype(src.dtypes[0])
        assert numpy.issubdtype(dtype, numpy.integer) # sanity check: int only
        nodata = src.nodata
        if nodata is None:
            nodata = -1 # default value
        assert abs(nodata - round(nodata)) < 1e-9 # sanity check: nodata is int

    # read through all windows via multiprocessing
    jobs = [(_read_window, (k, v, domain_fpath), {}) for k, v in _grid.items()]
    results: list[alias.RasterTile] = utils.ParallelExecutor().run(jobs)

    # gather unique values (exclude nodata)
    unique_values = set()
    for (_, arr) in results:
        unique_values.update(numpy.unique(arr))
    if nodata in unique_values:
        unique_values.remove(nodata) # remove nodata if present
    # safety check against empty input raster
    if not unique_values:
        raise ValueError("No valid data values found.")
    # make sure minimal aligns with index base
    if min(unique_values) != index_base:
        raise ValueError(f'Min {min(unique_values)} != base {index_base}')

    # global mapping: raw i..K  ->  0..K-1 ----
    remap = numpy.array(sorted(unique_values), dtype=numpy.int64)
    # Map each block: valid raw -> index in [0..K-1], nodata -> -1
    for (_, arr) in results:
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

    # return converted domain mapping and the global max index
    return results, remap.size

def _read_window(
        window_id: tuple[int, int],
        raster_window: alias.RasterWindow,
        raster_fpath: str
    ) -> tuple[tuple[int, int], numpy.ndarray]:
    '''Read domain raster via a window and return the first channel.'''

    # read raster at window and return
    with rasterio.open(raster_fpath, 'r') as src:
        arr = src.read(1, window=raster_window) # [1, H, W] for 2D rasters
        arr = arr.astype(numpy.int64, copy=False) # ensure negatives support
        return window_id, arr

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

def is_valid(
    arr: numpy.ndarray,
    threshold: float
) -> bool:
    '''Return True if fraction of valid pixels >= threshold.'''

    valid = arr != -1
    if valid.size == 0:
        return False
    return float(valid.mean()) >= float(threshold)
