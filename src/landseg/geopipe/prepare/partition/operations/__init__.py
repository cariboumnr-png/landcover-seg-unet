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
Top-level namespace for `landseg.geopipe.prepare.partition.operations`.

Exposes data partitioning algorithms, AOI solvers, spatial filtering,
and hydration utilities via lazy resolution.

Public APIs:
    - `AoiSplitsResult`: Container for AOI partitioned block coords.
    - `HydrationResults`: Container for hydrated block coordinates.
    - `SplitsResult`: Container for split coordinates and class stats.
    - `filter_safe_tiles`: Filter candidate tiles to prevent overlap.
    - `hydrate_train_split`: Greedily hydrate training split.
    - `intersect_aoi_raster`: Find candidate blocks intersecting AOI.
    - `resolve_aoi_partitions`: Resolve splits with priority logic.
    - `score_blocks`: Score and rank candidate blocks.
    - `stratified_splitter`: Stratified splitter for train/val/test.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'AoiSplitsResult',
    'HydrationResults',
    'SplitsResult',
    # functions
    'filter_safe_tiles',
    'hydrate_train_split',
    'intersect_aoi_raster',
    'resolve_aoi_partitions',
    'score_blocks',
    'stratified_splitter',
]


# for static check
if typing.TYPE_CHECKING:
    from .aoi import (
        AoiSplitsResult,
        intersect_aoi_raster,
        resolve_aoi_partitions,
    )
    from .filter import (
        filter_safe_tiles,
    )
    from .hydrate import (
        HydrationResults,
        hydrate_train_split,
    )
    from .score import (
        score_blocks,
    )
    from .stratify import (
        SplitsResult,
        stratified_splitter,
    )


def __getattr__(name: str):
    if name in {
        'AoiSplitsResult',
        'intersect_aoi_raster',
        'resolve_aoi_partitions',
    }:
        obj = importlib.import_module('.aoi', __package__)
        return getattr(obj, name)

    if name in {'filter_safe_tiles'}:
        obj = importlib.import_module('.filter', __package__)
        return getattr(obj, name)

    if name in {
        'HydrationResults',
        'hydrate_train_split',
    }:
        obj = importlib.import_module('.hydrate', __package__)
        return getattr(obj, name)

    if name in {'score_blocks'}:
        obj = importlib.import_module('.score', __package__)
        return getattr(obj, name)

    if name in {
        'SplitsResult',
        'stratified_splitter',
    }:
        obj = importlib.import_module('.stratify', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
