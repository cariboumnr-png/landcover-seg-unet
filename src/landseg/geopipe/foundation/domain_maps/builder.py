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
Tools for preparing domain knowledge mapped onto a world grid.

This module provides the public API for constructing per-domain
DomainTileMap objects. It coordinates the following operations:

    - Align the provided world grid to each domain raster using the
        grid's pixel-offset calculation.
    - Load an existing DomainTileMap artifact if present, validating its
        schema and integrity.
    - Otherwise, build a new DomainTileMap by reading all grid windows
        from the categorical raster, filtering tiles by valid-pixel
        fraction, computing majority statistics, deriving normalized
        frequency vectors, and projecting them onto PCA components to
        reach a target explained variance.
    - Persist each new DomainTileMap as a JSON payload and a metadata
        sidecar including schema_id, context, hash, and grid association.

Configuration is supplied via a structured dictionary containing:
    - 'dirpath': directory with domain rasters.
    - 'files': list of domain raster entries ('name', 'index_base').
    - 'valid_threshold': minimum fraction of valid pixels for tile
        acceptance.
    - 'target_variance': PCA target cumulative explained variance.
    - 'output_dirpath': where DomainTileMap artifacts are written.

The output is a mapping from domain base names (filename without suffix)
to DomainTileMap instances, suitable for downstream model conditioning or
task-level feature assembly.
'''

# standard imports
import dataclasses
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.foundation.common.alias as alias
import landseg.utils as utils

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class _BuildingContext:
    '''doc'''
    max_index: int
    major_freq_mean: float
    major_freq_min: float
    pca_axes_n: int
    explained_variance: float
    valid_coords: list[tuple[int, int]]

# -------------------------------Public Function-------------------------------
def build_domain(
    grid_id: str,
    mapped_tiles: alias.RasterTileDict,
    valid_threshold: float,
    target_variance: float,
    logger: utils.Logger,
) -> geo_core.DomainTileMap:
    '''doc'''

    # get domain tiles dict
    context, domain_dict = _get_domain_dict(
        mapped_tiles,
        valid_threshold,
        target_variance,
        logger
    )
    # instantiate class object from dict
    domain_map = geo_core.DomainTileMap.from_dict(domain_dict)

    # update metadata and return
    domain_map.meta.update({
        'world_grid_ids': [grid_id],
        'valid_threshold': valid_threshold,
        'target_variance': target_variance,
        'max_index': context.max_index,
        'major_freq_mean': context.major_freq_mean,
        'major_freq_min': context.major_freq_min,
        'pca_axes_n': context.pca_axes_n,
        'explained_variance': context.explained_variance,
    })
    return domain_map

# ------------------------------private  function------------------------------
def _get_domain_dict(
    raster_tiles: alias.RasterTileDict,
    valid_threshold: float,
    target_variance: float,
    logger: utils.Logger
) -> tuple[_BuildingContext, dict[tuple[int, int], geo_core.DomainTile]]:
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

    # prep
    domain_dict: dict[tuple[int, int], geo_core.DomainTile] = {}
    context = _BuildingContext(
        max_index = int(raster_tiles.pop((-999, -999)).flat[0]), # pop
        major_freq_mean=0.0,
        major_freq_min=1.0,
        pca_axes_n=0,
        explained_variance=0,
        valid_coords=[]
    )

    # first iteration to get indices of valid tiles and index range
    logger.log('INFO', 'Filter input raster tiles')
    for c, tile in raster_tiles.items():
        if c in domain_dict: # skip keys that already in data
            continue
        # add all to _data
        domain_dict[c] = {
            'majority': None,
            'major_freq': None,
            'pca_feature': None,
        }
        if _is_valid(tile, valid_threshold):
            # add to valid coordinates list
            context.valid_coords.append(c)

    # get majority index for valid tiles - calc here
    logger.log('INFO', 'Calculate majority class from new tiles')
    for c in set(raster_tiles.keys()):
        if all(domain_dict[c].values()): # skip keys that already have data
            continue
        tile = raster_tiles[c]
        values, counts = numpy.unique(tile, return_counts=True)
        # update domain tile dict
        major = values[numpy.argmax(counts)].item() # serializable
        freq = counts[numpy.argmax(counts)] / sum(counts)
        domain_dict[c].update({'majority': major, 'major_freq': freq})
        # update major_freq stats
        context.major_freq_min = min(context.major_freq_min, freq)
        context.major_freq_mean += freq
    context.major_freq_mean /= len(context.valid_coords)

    # get pca transform for valid tiles - calc delegated to transform.py
    logger.log('INFO', 'PCA transforming all valid tiles')
    freqs: dict[tuple[int, int], numpy.ndarray] = {}
    for c in context.valid_coords: # calculate on all valid tiles
        tile = raster_tiles[c]
        freqs[c] = _norm_freq(tile, (0, context.max_index)) # 0-based
    # get full pca
    z, context.explained_variance, context.pca_axes_n = _pca_transform(
        freqs, target_variance)
    # assign to each valid tile
    for c in context.valid_coords:
        tile = raster_tiles[c]
        domain_dict[c].update({'pca_feature': [float(x) for x in z[c]]})

    # return only valid tiles
    valid_set = set(context.valid_coords)
    return context, {c: v for c, v in domain_dict.items() if c in valid_set}

def _is_valid(
    arr: numpy.ndarray,
    threshold: float
) -> bool:
    valid = arr != -1
    if valid.size == 0:
        return False
    return float(valid.mean()) >= float(threshold) # valid px% >= threshold

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

def _pca_transform(
    freqs: dict[tuple[int, int], numpy.ndarray],
    target_var: float
) -> tuple[dict[tuple[int, int], numpy.ndarray], float, int]:
    '''
    Project tile class-frequency vectors onto PCA axes to reach variance.

    Args:
        freqs: Mapping from tile coordinates (e.g., (x, y)) to 1-D class-
            frequency vectors of length K (non-negative; sum to 1).
        target_var: Target cumulative explained variance in (0, 1].

    Returns:
        tuple:
        - Dict mapping the same tile coordinates to PCA vectors of length
        k, where k is the smallest number of components whose cumulative
        explained variance ≥ target_var. Each vector is dtype float32.
        - The cumulative explained variance captured by the selected k
        components, expressed in percent (e.g., 92.34).

    Notes:
    - PCA is fit once across all provided tiles, and the top-k components
        are applied to each tile.
    - If target_var is very small or the spectrum is dominated by PC1, k
    may be 1.
    '''

    # lock ordering in one pass
    items = list(freqs.items())
    keys = [k for k, _ in items]
    rows = [v for _, v in items]

    # (N, D) stack
    freq_stack = numpy.stack(rows, axis=0)

    # fit once, then choose k
    mean, components_full, evr_full = _fit_pca(freq_stack)
    k = _k_from_target_evr(evr_full, target_var)

    # slice the first k components and explained variance
    components = components_full[:k, :]
    evr_k = evr_full[:k]

    # project
    z = _transform(freq_stack, mean, components)  # (N, k)

    # map z back to freqs keys
    mapped_z = dict(zip(keys, z))
    return mapped_z, float(evr_k.sum() * 100.0), k

def _fit_pca(x: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    '''
    Fit PCA on rows of X (N x D) using SVD.
    Returns mean (D,), components_full (D x D_or_N), evr_full (L,).
    '''
    assert x.ndim == 2, x.shape
    # center
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # economical SVD on centered data
    _, s, vt = numpy.linalg.svd(xc, full_matrices=False) # Xc = U S V^T
    # vt shape: (L, D) where L = min(N, D)
    var = (s ** 2) / (x.shape[0] - 1) # per-PC variance (L,)
    total_var = (xc ** 2).sum() / (x.shape[0] - 1)
    evr = var / total_var
    return mean.squeeze(0), vt, evr

def _k_from_target_evr(
    evr_full: numpy.ndarray,
    target: float
) -> int:
    '''
    Return the smallest k s.t. cumulative explained variance >= target.
    target is in [0, 1], e.g., 0.95 for 95%.
    '''

    if not 0.0 < target <= 1.0:
        raise ValueError('target must be in (0, 1].')
    # if not numpy.all(numpy.isfinite(evr_full)):
    #     print('DEBUG: evr_full has non-finite values:', evr_full)
    cum = numpy.cumsum(evr_full)
    # print(f'target={target}, first_evr={evr_full[0]}, cum_last={cum[-1]}')
    k = int(numpy.searchsorted(cum, target, side='left') + 1)
    k = max(1, min(k, evr_full.shape[0]))
    return k

def _transform(
        x: numpy.ndarray,
        mean: numpy.ndarray,
        components: numpy.ndarray
    ) -> numpy.ndarray:
    '''Project rows of X onto PCA components (k x D).'''

    xc = x - mean
    z = xc @ components.T  # (N, k)
    z = numpy.asarray(z, dtype=numpy.float32) # ensure float32
    z = numpy.atleast_2d(z) # ensure 2D shape
    return z
