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
Top-level namespace for `landseg.geopipe.ingest`.

Coordinates the ingestion of harmonized geospatial rasters into
tiled domain maps and canonical data blocks, providing logging,
context resolution, and pipeline execution tools via lazy module
resolution.

Public APIs:
    - `BlockBuildingParameters`: Config for block pipeline.
    - `DomainBuildingParameters`: Config for domain mapping.
    - `IngestionContext`: Container holding resolved grid and rasters.
    - `IngestionLogger`: Structured logger for ingestion stages.
    - `build_ingestion_context`: Load ingestion context from report.
    - `prepare_domain_maps`: Generates domain tilemaps from rasters.
    - `run_blocks_building`: Runs canonical data block pipeline.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuildingParameters',
    'DomainBuildingParameters',
    'IngestionContext',
    'IngestionLogger',
    # functions
    'build_ingestion_context',
    'prepare_domain_maps',
    'run_blocks_building',
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .context import (
        IngestionContext,
        build_ingestion_context,
    )
    from .data_blocks import (
        BlockBuildingParameters,
        run_blocks_building,
    )
    from .domain_maps import (
        DomainBuildingParameters,
        prepare_domain_maps,
    )
    from .logger import IngestionLogger


def __getattr__(name: str):
    if name in {'IngestionContext', 'build_ingestion_context'}:
        return getattr(
            importlib.import_module('.context', __package__), name
        )

    if name in {'IngestionLogger'}:
        return getattr(
            importlib.import_module('.logger', __package__), name
        )

    if name in {'BlockBuildingParameters', 'run_blocks_building'}:
        return getattr(
            importlib.import_module('.data_blocks', __package__), name
        )

    if name in {'DomainBuildingParameters', 'prepare_domain_maps'}:
        return getattr(
            importlib.import_module('.domain_maps', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
