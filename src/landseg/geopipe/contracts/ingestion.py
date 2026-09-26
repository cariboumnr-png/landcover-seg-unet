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

# pylint: disable=missing-function-docstring

'''
TypedDict definitions for data ingestion execution summaries and
reports.

This module provides schemas for serializing execution reports, domain
statistics, block generation metrics, and catalog update summaries.

Public APIs:
    - `CollisionPolicyType`: Literal type for collision policies.
    - `CollisionRecord`: TypedDict for block collision entry.
    - `RunCollisionManifest`: TypedDict for run-level collision report.
    - `CollisionStats`: TypedDict for aggregate collision metrics.
    - `DomainStats`: TypedDict for domain layer re-indexing statistics.
    - `DomainMapReport`: TypedDict for domain map execution report.
    - `BlockStats`: TypedDict for data block mapping and build stats.
    - `ManifestStats`: TypedDict for catalog/schema update details.
    - `DataBlocksReport`: TypedDict for data block execution report.
    - `IngestionPipelineConfig`: Protocol for pipeline configs.
    - `IngestReportSchema`: TypedDict for root data ingestion report.
    - `IngestionRunRecord`: TypedDict for ingestion run manifest entry.
    - `IngestionRunManifest`: Type alias for runs ledger mapping.
'''

# standard imports
from __future__ import annotations
import typing

# ----- public types
CollisionPolicyType = typing.Literal['skip', 'overwrite', 'error']


class IngestionPipelineConfig(typing.Protocol):
    '''Shape of the Ingestion pipeline configurations.'''
    @property
    def domains(self) -> _DomainsConfig: ...
    @property
    def datablocks(self) -> _DataBlocksConfig: ...


class _DomainsConfig(typing.Protocol):
    @property
    def valid_threshold(self) -> float: ...
    @property
    def target_variance(self) -> float: ...


class _DataBlocksConfig(typing.Protocol):
    @property
    def ignore_index(self) -> int: ...
    @property
    def image_dem_pad(self) -> int: ...
    @property
    def add_topo(self) -> list[str] | None: ...
    @property
    def add_spectral(self) -> list[str] | None: ...
    @property
    def collision_policy(self) -> CollisionPolicyType: ...


class CollisionRecord(typing.TypedDict):
    '''Record of an individual block spatial collision and action taken.'''
    block_name: str
    grid_coord: list[int]
    incumbent_ingest_run: str | None
    incumbent_harmonize_run: str | None
    action_taken: typing.Literal['skipped', 'overwritten', 'error']


class RunCollisionManifest(typing.TypedDict):
    '''Run-level collision audit manifest persisted as collisions.json.'''
    ingestion_run_id: str
    ingestion_run_uid: str
    harmonization_run_id: str
    collision_policy: CollisionPolicyType
    total_collided: int
    collided_blocks: list[CollisionRecord]


class CollisionStats(typing.TypedDict):
    '''Aggregate metrics on intra-pool spatial block collisions.'''
    blocks_candidate: int
    blocks_collided: int
    blocks_skipped: int
    blocks_overwritten: int
    blocks_added: int


class IngestionRunRecord(typing.TypedDict):
    '''Single ingestion run entry stored in runs manifest.'''
    run_uid: str
    run_id: str
    harmonization_run_uid: str
    harmonization_run_id: str
    status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    timestamp: str
    fingerprint: str
    run_folder: str


class IngestReportSchema(typing.TypedDict):
    '''Root report mapping the entire data ingestion pipeline run.'''
    run_uid: str
    run_id: str
    harmonization_run_uid: str
    harmonization_run_id: str
    timestamp: str
    fingerprint: str
    status: typing.Literal['SUCCESS', 'FAILED', 'SKIPPED']
    domain_maps: list[DomainMapReport]
    data_blocks: DataBlocksReport | None


class DomainMapReport(typing.TypedDict):
    '''Execution report for domain map preparation.'''
    name: str
    status: typing.Literal['loaded', 'created']
    input_filepath: str
    domain_filepath: str
    tiles_filepath: str
    duration_sec: float
    stats: DomainStats | None


class DomainStats(typing.TypedDict):
    '''Re-indexing and mapping statistics for a domain layer.'''
    max_index: int
    valid_coords_count: int
    major_freq_mean: float
    major_freq_min: float
    pca_axes_n: int
    explained_variance: float


class DataBlocksReport(typing.TypedDict):
    '''
    Execution report for data block partitioning (dev or test holdout).
    '''
    image_filepath: str
    label_filepath: str | None
    duration_sec: float
    stats: BlockStats | None
    manifest: ManifestStats | None
    collisions: CollisionStats | None


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
