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
TypedDict definitions for data harmonization execution summaries/reports.

This module provides schemas for serializing execution reports, source
provenance, and grid metadata for data harmonization runs.

Public APIs:
    - `ProvenanceRecord`: TypedDict for raw raster file provenance.
    - `WorldGridReport`: TypedDict for world grid summary report.
    - `HarmonizationReportSchema`: TypedDict for overall pipeline run report.
'''

# standard imports
from __future__ import annotations
import typing


# ----- public types
class HarmonizationReportSchema(typing.TypedDict):
    '''Root report mapping the entire data harmonization pipeline run.'''
    run_id: str
    timestamp: str
    status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    provenance: dict[str, ProvenanceRecord]
    harmonized_sources: dict[str, str]
    finalized_rasters: dict[str, str]
    valid_mask_raster: str
    world_grid: WorldGridReport | None

    
class ProvenanceRecord(typing.TypedDict):
    '''Provenance record for a raw source raster file.'''
    path: str
    size_bytes: int
    mtime: float


class WorldGridReport(typing.TypedDict):
    '''Summary report for a generated world grid layout.'''
    grid_fpath: str
    grid_id: str
    crs: str
    pixel_size: tuple[float, float]
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]
