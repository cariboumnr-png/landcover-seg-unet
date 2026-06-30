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
TypedDict definitions for data ingestion execution summaries/reports.
'''

from __future__ import annotations
import typing

class WorldGridReport(typing.TypedDict):
    '''Execution report for world grid preparation.'''
    grid_id: str
    status: typing.Literal['loaded', 'created_and_loaded']
    grid_filepath: str
    crs: str
    pixel_size: tuple[float, float]
    tile_size: tuple[int, int]
    tile_overlap: int
    duration_sec: float

class DomainStats(typing.TypedDict):
    '''Re-indexing and mapping statistics for a domain layer.'''
    max_index: int
    valid_coords_count: int
    major_freq_mean: float
    major_freq_min: float
    pca_axes_n: int
    explained_variance: float

class DomainMapReport(typing.TypedDict):
    '''Execution report for domain map preparation.'''
    name: str
    status: typing.Literal['loaded', 'created']
    input_filepath: str
    domain_filepath: str
    tiles_filepath: str
    duration_sec: float
    stats: DomainStats | None

class BlockStats(typing.TypedDict):
    '''Statistics for raster window mapping and data block builds.'''
    shared_raster_windows: int
    expected_shape_windows: int
    blocks_on_disk_before: int
    blocks_to_process: int
    damaged_blocks_removed: int
    blocks_created: int

class ManifestStats(typing.TypedDict):
    '''Data blocks catalog and schema update details.'''
    catalog_status: str
    cataloged_blocks_count: int
    catalog_updated: bool
    schema_updated: bool

class DataBlocksReport(typing.TypedDict):
    '''Execution report for data block partitioning (dev or test holdout).'''
    status: typing.Literal['built_and_updated', 'skipped']
    image_filepath: str
    label_filepath: str | None
    duration_sec: float
    stats: BlockStats | None
    manifest: ManifestStats | None

class IngestReportSchema(typing.TypedDict):
    '''Root report mapping the entire data ingestion pipeline run.'''
    run_id: str
    timestamp: str
    status: typing.Literal['SUCCESS', 'FAILED']
    world_grid: WorldGridReport | None
    domain_maps: list[DomainMapReport]
    data_blocks: dict[typing.Literal['dev', 'test'], DataBlocksReport]
