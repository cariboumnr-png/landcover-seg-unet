# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Geographic Area of Interest (AOI) selection and conflict resolution.

Provides spatial intersection routines to select candidate data blocks
from external AOI GeoTIFF rasters with coordinate reference system (CRS)
reprojection, resolving multi-split overlap conflicts under a strict
priority order (test > val > train).

Public APIs:
    - AoiSplitsResult: container for AOI-partitioned block coordinates.
    - intersect_aoi_raster: find candidate blocks intersecting an AOI.
    - resolve_aoi_partitions: resolve splits with priority logic.
'''

# standard imports
import dataclasses
import os
# third-party imports
import rasterio
import rasterio.transform
import rasterio.warp
# local imports
import landseg.geopipe.prepare as prepare


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class AoiSplitsResult:
    '''Container for AOI-partitioned block coordinate sets.'''
    test: list[tuple[int, int]]
    val: list[tuple[int, int]]
    train: list[tuple[int, int]]
    unassigned: list[tuple[int, int]]


# ----- public functions
def intersect_aoi_raster(
    aoi_path: str,
    candidate_blocks: list[tuple[int, int]],
    *,
    block_size: tuple[int, int],
    canvas_crs: str,
    canvas_transform: rasterio.transform.Affine,
    min_overlap: float = 0.5,
) -> list[tuple[int, int]]:
    '''
    Find candidate block coordinates intersecting an AOI raster.

    Args:
        aoi_path:
            file path to the AOI GeoTIFF raster.
        candidate_blocks:
            list of candidate top-left (row, col) pixel coordinates.
        block_size:
            block spatial dimensions (size_row, size_col).
        canvas_crs:
            coordinate reference system of canonical grid canvas.
        canvas_transform:
            affine transform of canonical grid canvas.
        min_overlap:
            minimum overlap ratio required to select a block.

    Returns:
        list[tuple[int, int]]:
            block coordinates matching the AOI footprint.
    '''
    aoi_bounds = _load_aoi_bounds(aoi_path, canvas_crs)

    matched: list[tuple[int, int]] = []
    for rc in candidate_blocks:
        if _check_overlap(
            rc, block_size, canvas_transform, aoi_bounds, min_overlap
        ):
            matched.append(rc)

    return matched


def resolve_aoi_partitions(
    candidate_blocks: list[tuple[int, int]],
    *,
    train_aoi: str | None = None,
    val_aoi: str | None = None,
    test_aoi: str | None = None,
    block_size: tuple[int, int],
    canvas_crs: str,
    canvas_transform: rasterio.transform.Affine,
    min_overlap: float = 0.5,
    logger: prepare.PreparationLogger | None = None,
) -> AoiSplitsResult:
    '''
    Resolve split partitions from AOI rasters with priority resolution.

    Priority for resolving multi-AOI conflicts: test > val > train.
    Emits warning messages when overlapping claims are resolved.

    Args:
        candidate_blocks:
            all valid candidate top-left (row, col) block tuples.
        train_aoi:
            optional file path to training AOI GeoTIFF.
        val_aoi:
            optional file path to validation AOI GeoTIFF.
        test_aoi:
            optional file path to test AOI GeoTIFF.
        block_size:
            block spatial dimensions (size_row, size_col).
        canvas_crs:
            coordinate reference system of canvas.
        canvas_transform:
            affine transform of canvas.
        min_overlap:
            minimum overlap ratio required for selection.
        logger:
            optional logger for emitting conflict warnings.

    Returns:
        AoiSplitsResult:
            resolved test, val, train, and unassigned block coordinates.
    '''
    test_raw = (
        intersect_aoi_raster(
            test_aoi,
            candidate_blocks,
            block_size=block_size,
            canvas_crs=canvas_crs,
            canvas_transform=canvas_transform,
            min_overlap=min_overlap,
        )
        if test_aoi
        else []
    )

    val_raw = (
        intersect_aoi_raster(
            val_aoi,
            candidate_blocks,
            block_size=block_size,
            canvas_crs=canvas_crs,
            canvas_transform=canvas_transform,
            min_overlap=min_overlap,
        )
        if val_aoi
        else []
    )

    train_raw = (
        intersect_aoi_raster(
            train_aoi,
            candidate_blocks,
            block_size=block_size,
            canvas_crs=canvas_crs,
            canvas_transform=canvas_transform,
            min_overlap=min_overlap,
        )
        if train_aoi
        else []
    )

    assigned_test = set(test_raw)
    assigned_val, assigned_train = _resolve_conflicts(
        val_raw, train_raw, assigned_test, logger
    )

    all_assigned = assigned_test | assigned_val | assigned_train
    unassigned = [b for b in candidate_blocks if b not in all_assigned]

    return AoiSplitsResult(
        test=list(assigned_test),
        val=list(assigned_val),
        train=list(assigned_train),
        unassigned=unassigned,
    )


# ----- private helpers
def _load_aoi_bounds(
    aoi_path: str,
    canvas_crs: str,
) -> tuple[float, float, float, float]:
    '''Load AOI raster bounds and reproject to canvas CRS if needed.'''
    if not os.path.exists(aoi_path):
        raise FileNotFoundError(f'AOI raster file not found: {aoi_path}')

    with rasterio.open(aoi_path) as src:
        aoi_crs = src.crs.to_string() if src.crs else canvas_crs
        left, bottom, right, top = src.bounds

    if aoi_crs != canvas_crs:
        left, bottom, right, top = rasterio.warp.transform_bounds(
            aoi_crs, canvas_crs, left, bottom, right, top
        )

    return (left, bottom, right, top)


def _check_overlap(
    row_col: tuple[int, int],
    block_size: tuple[int, int],
    transform: rasterio.transform.Affine,
    aoi_bounds: tuple[float, float, float, float],
    min_overlap: float,
) -> bool:
    '''Calculate if a block exceeds the minimum overlap threshold.'''
    row, col = row_col
    left, bottom, right, top = aoi_bounds
    res_x = abs(transform.a)
    res_y = abs(transform.e)

    b_min_x = transform.c + col * transform.a
    b_max_x = b_min_x + block_size[1] * res_x
    b_max_y = transform.f + row * transform.e
    b_min_y = b_max_y - block_size[0] * res_y

    x0, x1 = min(b_min_x, b_max_x), max(b_min_x, b_max_x)
    y0, y1 = min(b_min_y, b_max_y), max(b_min_y, b_max_y)

    i_min_x = max(x0, left)
    i_max_x = min(x1, right)
    i_min_y = max(y0, bottom)
    i_max_y = min(y1, top)

    if i_max_x > i_min_x and i_max_y > i_min_y:
        intersection = (i_max_x - i_min_x) * (i_max_y - i_min_y)
        block_area = (x1 - x0) * (y1 - y0)
        return block_area > 0 and (intersection / block_area) >= min_overlap

    return False


def _resolve_conflicts(
    val_raw: list[tuple[int, int]],
    train_raw: list[tuple[int, int]],
    assigned_test: set[tuple[int, int]],
    logger: prepare.PreparationLogger | None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    '''Resolve overlap conflicts using priority: test > val > train.'''
    assigned_val: set[tuple[int, int]] = set()
    assigned_train: set[tuple[int, int]] = set()

    for b in val_raw:
        if b in assigned_test:
            if logger is not None:
                logger.log(
                    'WARNING',
                    f'Block {b} overlaps both test and validation AOIs. '
                    'Assigned to test by priority.'
                )
        else:
            assigned_val.add(b)

    for b in train_raw:
        if b in assigned_test:
            if logger is not None:
                logger.log(
                    'WARNING',
                    f'Block {b} overlaps both test and training AOIs. '
                    'Assigned to test by priority.'
                )
        elif b in assigned_val:
            if logger is not None:
                logger.log(
                    'WARNING',
                    f'Block {b} overlaps both validation and training AOIs. '
                    'Assigned to validation by priority.'
                )
        else:
            assigned_train.add(b)

    return assigned_val, assigned_train
