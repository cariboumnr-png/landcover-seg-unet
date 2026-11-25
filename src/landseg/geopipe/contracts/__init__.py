# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Top-level namespace for `landseg.geopipe.contracts`.

Public APIs:
    - `BlocksBuildingStats`: Data block build statistics.
    - `BlocksPartition`: TypedDict mapping block IDs across splits.
    - `CollisionPolicyType`: Literal type for block collision policies.
    - `CollisionRecord`: Record of block collision and action taken.
    - `DataBlocksReport`: Execution report for data block partitioning.
    - `DataPartitionReport`: Report for dataset splitting.
    - `DomainMapReport`: Execution report for domain map preparation.
    - `DomainStats`: Re-indexing statistics for a domain layer.
    - `GridReportSchema`: Execution report for world grid pipeline.
    - `HarmonizationPipelineConfig`: Harmonization pipeline config.
    - `HarmonizationReportSchema`: Report schema for harmonization.
    - `HarmonizationRunRecord`: Single harmonization run manifest entry.
    - `ImageBandStats`: TypedDict for image band statistics.
    - `IngestReportSchema`: Root report for data ingestion pipeline.
    - `IngestionPipelineConfig`: Ingestion pipeline config shape.
    - `IngestionRunRecord`: Single ingestion run manifest entry.
    - `ManifestStats`: Catalog and schema update details.
    - `MappedRasterWindowsPayload`: Serialized raster window map.
    - `NormalizationReport`: Report for block materialization.
    - `PREPARED_SCHEMA_ID`: Constant string for prepared schema ID.
    - `PartitionSummary`: Summary of raw splits and hydration.
    - `PreparationDatasetViewConfig`: Preparation dataset view config.
    - `PreparationPipelineConfig`: Preparation pipeline config shape.
    - `PreparationReportSchema`: Root summary schema for prepare runs.
    - `PreparedSchema`: TypedDict for dataset preparation schema.
    - `ProvenanceRecord`: Provenance record for raw source raster.
    - `RunCollisionManifest`: Collision audit manifest for ingest runs.
    - `SchemaReport`: Report for dataset schema generation.
    - `TargetHeadsSchema`: TypedDict for target heads hierarchy.
    - `WorldGridPrepConfig`: Config shape to prepare world grid.
    - `WorldGridReport`: Summary report for world grid layout.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'BlocksBuildingStats',
    'BlocksPartition',
    'CollisionPolicyType',
    'CollisionRecord',
    'DataBlocksReport',
    'DataPartitionReport',
    'DomainMapReport',
    'DomainStats',
    'GridReportSchema',
    'HarmonizationPipelineConfig',
    'HarmonizationReportSchema',
    'HarmonizationRunRecord',
    'ImageBandStats',
    'IngestReportSchema',
    'IngestionPipelineConfig',
    'IngestionRunRecord',
    'ManifestStats',
    'MappedRasterWindowsPayload',
    'NormalizationReport',
    'PREPARED_SCHEMA_ID',
    'PartitionSummary',
    'PreparationDatasetViewConfig',
    'PreparationPipelineConfig',
    'PreparationReportSchema',
    'PreparedSchema',
    'ProvenanceRecord',
    'RunCollisionManifest',
    'SchemaReport',
    'TargetHeadsSchema',
    'WorldGridPrepConfig',
    'WorldGridReport',
]


# for static check
if typing.TYPE_CHECKING:
    from .grid import (
        GridReportSchema,
        WorldGridPrepConfig,
        WorldGridReport,
    )
    from .harmonization import (
        HarmonizationPipelineConfig,
        HarmonizationReportSchema,
        HarmonizationRunRecord,
        ProvenanceRecord,
    )
    from .ingestion import (
        BlocksBuildingStats,
        CollisionPolicyType,
        CollisionRecord,
        DataBlocksReport,
        DomainMapReport,
        DomainStats,
        IngestReportSchema,
        IngestionPipelineConfig,
        IngestionRunRecord,
        ManifestStats,
        MappedRasterWindowsPayload,
        RunCollisionManifest,
    )
    from .preparation import (
        BlocksPartition,
        DataPartitionReport,
        ImageBandStats,
        NormalizationReport,
        PREPARED_SCHEMA_ID,
        PartitionSummary,
        PreparationDatasetViewConfig,
        PreparationPipelineConfig,
        PreparationReportSchema,
        PreparedSchema,
        SchemaReport,
        TargetHeadsSchema,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.grid': [
        'GridReportSchema',
        'WorldGridPrepConfig',
        'WorldGridReport',
    ],
    '.harmonization': [
        'HarmonizationPipelineConfig',
        'HarmonizationReportSchema',
        'HarmonizationRunRecord',
        'ProvenanceRecord',
    ],
    '.ingestion': [
        'BlocksBuildingStats',
        'CollisionPolicyType',
        'CollisionRecord',
        'DataBlocksReport',
        'DomainMapReport',
        'DomainStats',
        'IngestReportSchema',
        'IngestionPipelineConfig',
        'IngestionRunRecord',
        'ManifestStats',
        'MappedRasterWindowsPayload',
        'RunCollisionManifest',
    ],
    '.preparation': [
        'BlocksPartition',
        'DataPartitionReport',
        'ImageBandStats',
        'NormalizationReport',
        'PREPARED_SCHEMA_ID',
        'PartitionSummary',
        'PreparationDatasetViewConfig',
        'PreparationPipelineConfig',
        'PreparationReportSchema',
        'PreparedSchema',
        'SchemaReport',
        'TargetHeadsSchema',
    ],
})
