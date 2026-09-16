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
harmonization adapters, and pipeline execution tools via lazy module
resolution.

Public APIs:
    - BlockBuildingParameters: Config for block pipeline.
    - DomainBuildingParameters: Config for domain mapping.
    - HarmonizedRasters: Container for harmonized raster paths.
    - IngestionLogger: Structured logger for ingestion stages.
    - prepare_domain_maps: Generates domain tilemaps from rasters.
    - read_harmonization_report: Extracts rasters from report.
    - run_blocks_building: Runs canonical data block pipeline.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuildingParameters',
    'DomainBuildingParameters',
    'IngestionLogger',
    'HarmonizedRasters',
    # functions
    'read_harmonization_report',
    'prepare_domain_maps',
    'run_blocks_building',
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .adapter import HarmonizedRasters, read_harmonization_report
    from .common import IngestionLogger
    from .data_blocks import BlockBuildingParameters, run_blocks_building
    from .domain_maps import DomainBuildingParameters, prepare_domain_maps


def __getattr__(name: str):
    if name in {'HarmonizedRasters', 'read_harmonization_report'}:
        return getattr(
            importlib.import_module('.adapter', __package__), name
        )

    if name in {'IngestionLogger'}:
        return getattr(
            importlib.import_module('.common', __package__), name
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
