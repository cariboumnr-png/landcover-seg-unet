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
Domain raster mapping utilities.

Reads a categorical domain raster, aligns it to a pre-built world grid,
and remaps label values into a compact, zero-based index space suitable
for downstream domain-feature constructio
'''

# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.geopipe.utils as geo_utils
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def map_domain_to_grid(
    world_grid: geo_core.GridLayout,
    raster_path: str,
    *,
    index_base: int,
    logger: utils.Logger,
) -> alias.RasterTileDict:
    '''
    Map a domain raster onto a world grid and re-index labels.

    The raster is read window-by-window over all grid tiles after
    aligning the grid to the raster via an integer pixel offset. All
    unique, non-nodata label values are collected globally and remapped
    to a contiguous index range starting at zero.

    Behavior:
    - Align the world grid to the raster using pixel-exact offsets.
    - Discover all unique label values excluding nodata.
    - Validate that the minimum label equals `index_base`.
    - Remap valid labels to the range [0 .. K-1]; assign -1 to nodata.

    Args:
        grid: World grid layout defining raster windows.
        raster_path: Path to the categorical domain raster.
        logger: Logger for progress reporting.
        index_base: Expected minimum value of valid labels in the raster.

    Returns:
        _DomainTilesPackage containing re-indexed tiles and grid metadata.
    '''

    logger.log('INFO', 'Mapping domain onto input world grid')
    # read domain raster and get arrays indexed to the grid tiles
    tiles, nodata = _read_raster(world_grid, raster_path)

    logger.log('INFO', 'Generating index mapping')
    # global mapping: raw i..K  ->  0..K-1 ----
    idx_map = _get_index_mapping(tiles, nodata, index_base)

    logger.log('INFO', 'Re-indexing domain to [0...k-1]')
    # Map each block: valid raw -> index in [0..K-1], nodata -> -1
    _tiles = _re_index(tiles, nodata, idx_map)

    # encode the max index (k-1) to the return dict for later quick access
    shape = world_grid.tile_size
    max_idx = idx_map.size - 1
    _tiles[(-999, -999)] = numpy.full(shape, max_idx, dtype=numpy.int16)

    logger.log('INFO', 'Domain mapped onto input world grid')
    return _tiles

# ------------------------------private  function------------------------------
def _read_raster(
    grid: geo_core.GridLayout,
    fpath: str,
) -> tuple[alias.RasterTileDict, int]:
    '''Read a raster over all grid windows.'''

    # open domain raster
    with geo_utils.open_rasters(fpath) as (src,):
        assert src
        # offset the world grid to align with input raster
        grid.offset_from(src)
        # infer dtype and nodata
        dtype = numpy.dtype(src.dtypes[0])
        assert numpy.issubdtype(dtype, numpy.integer) # sanity check: int only
        nodata = src.nodata
        if nodata is None:
            nodata = -1 # default value
        assert abs(nodata - round(nodata)) < 1e-9 # sanity check: nodata is int

    # read through all windows via multiprocessing
    jobs = [(_read, (k, v, fpath, grid.tile_size), {})for k, v in grid.items()]
    all_tiles: list[alias.RasterTile] = utils.ParallelExecutor().run(jobs)
    all_tiles = [(_, t) for (_, t) in all_tiles if t.sum() != 0] # filter
    return dict(all_tiles), nodata

def _read(
    raster_window_id: tuple[int, int],
    raster_window: alias.RasterWindow,
    raster_fpath: str,
    expected_h_w: tuple[int, int]
) -> alias.RasterTile:
    '''Read a single raster window and return its first band.'''

    # if arr is not of expected H, W return an empty array
    if not (raster_window.height, raster_window.width) == expected_h_w:
        return raster_window_id, numpy.array([])

    # read raster at window and return
    with geo_utils.open_rasters(raster_fpath) as (src,):
        assert src, f'Invalid domain raster source: {raster_fpath}'
        arr = src.read(1, window=raster_window, boundless=True) # [1, H, W]
        arr = arr.astype(numpy.int16, copy=False) # avoids OOM
        return raster_window_id, arr

def _get_index_mapping(
    tiles: alias.RasterTileDict,
    nodata: int,
    index_base: int
) -> numpy.ndarray:
    '''Compute a global, sorted label remapping excluding nodata.'''

    # iteration on all tiles to gather unique values (exclude nodata)
    unique_values = set()
    for arr in tiles.values():
        unique_values.update(numpy.unique(arr))
    if nodata in unique_values:
        unique_values.remove(nodata) # remove nodata if present
    # safety check against empty input raster
    if not unique_values:
        raise ValueError("No valid data values found.")
    # make sure minimal aligns with index base
    if min(unique_values) != index_base:
        raise ValueError(
            f'Min value {min(unique_values)} != base {index_base}'
        )
    # global mapping: raw i..K  ->  0..K-1 ----
    mapping = numpy.array(sorted(unique_values), dtype=numpy.int64)
    return mapping

def _re_index(
    tiles: alias.RasterTileDict,
    nodata: int,
    mapping: numpy.ndarray
) -> alias.RasterTileDict:
    '''Apply a global index remapping to all raster tiles in-place.'''

    for arr in tiles.values():
        mask_valid = arr != nodata
        # skip if a grid tile does not contain any data
        if not numpy.any(mask_valid):
            arr[...] = -1 # ensure to -1
            continue
        # get indices of insertion
        idx = numpy.searchsorted(mapping, arr[mask_valid])
        # safety checks
        if not numpy.all(mapping[idx] == arr[mask_valid]):
            raise ValueError('Encountered value not present in remap domain')
        if numpy.any(idx >= mapping.size):
            raise ValueError('Encountered value outside remap domain')
        # assgin values to array
        arr[mask_valid] = idx
        arr[~mask_valid] = -1
    return tiles
