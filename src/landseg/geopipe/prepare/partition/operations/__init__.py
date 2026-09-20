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
    - AoiSplitsResult: container for AOI-partitioned block coordinates.
    - PartitionParameters: configuration for data block partitioning.
    - PartitionResults: container for partitioned splits and hydration.
    - SplitsResult: container for split coordinates and class stats.
    - HydrationResults: container for hydrated block coordinates.
    - create_blocks_partition: split blocks with spatial safety.
    - filter_safe_tiles: filter candidate tiles to prevent overlap.
    - hydrate_train_split: greedily hydrate training split.
    - intersect_aoi_raster: find candidate blocks intersecting an AOI.
    - resolve_aoi_partitions: resolve splits with priority logic.
    - score_blocks: score and rank candidate blocks.
    - stratified_splitter: stratified splitter for train/val/test.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'AoiSplitsResult',
    'SplitsResult',
    'HydrationResults',
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
    from .filter import filter_safe_tiles
    from .hydrate import HydrationResults, hydrate_train_split
    from .score import score_blocks
    from .stratify import SplitsResult, stratified_splitter


def __getattr__(name: str):

    if name in {
        'AoiSplitsResult',
        'intersect_aoi_raster',
        'resolve_aoi_partitions',
    }:
        return getattr(importlib.import_module('.aoi', __package__), name)

    if name in {'filter_safe_tiles'}:
        return getattr(importlib.import_module('.filter', __package__), name)

    if name in {'HydrationResults', 'hydrate_train_split'}:
        return getattr(importlib.import_module('.hydrate', __package__), name)

    if name in {'score_blocks'}:
        return getattr(importlib.import_module('.score', __package__), name)

    if name in {'SplitsResult', 'stratified_splitter'}:
        return getattr(importlib.import_module('.stratify', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
