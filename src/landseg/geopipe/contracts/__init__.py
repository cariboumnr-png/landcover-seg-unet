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
Top-level namespace for `landseg.geopipe.contracts`.

Public APIs:
    - `GridReportSchema`: Execution report for world grid pipeline.
    - `WorldGridReport`: Summary report for world grid layout.
    - `HarmonizationReportSchema`: Report schema for harmonization.
    - `ProvenanceRecord`: Provenance record for raw source raster.
    - `BlockStats`: Data block build statistics.
    - `DataBlocksReport`: Execution report for data block partitioning.
    - `DomainMapReport`: Execution report for domain map preparation.
    - `DomainStats`: Re-indexing statistics for a domain layer.
    - `IngestReportSchema`: Root report for data ingestion pipeline.
    - `ManifestStats`: Catalog and schema update details.
    - `BlocksPartition`: TypedDict mapping block IDs across splits.
    - `DataPartitionReport`: Report for dataset splitting.
    - `ImageBandStats`: TypedDict for image band statistics.
    - `NormalizationReport`: Report for block materialization.
    - `PartitionSummary`: Summary of raw splits and hydration.
    - `PREPARED_SCHEMA_ID`: Constant string for prepared schema ID.
    - `PreparationReportSchema`: Root summary schema for prepare runs.
    - `PreparedSchema`: TypedDict for dataset preparation schema.
    - `SchemaReport`: Report for dataset schema generation.
    - `TargetHeadsSchema`: TypedDict for target heads hierarchy.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # typing
    'BlockStats',
    'BlocksPartition',
    'DataBlocksReport',
    'DataPartitionReport',
    'DomainMapReport',
    'DomainStats',
    'GridReportSchema',
    'HarmonizationReportSchema',
    'HarmonizationRunRecord',
    'ImageBandStats',
    'IngestReportSchema',
    'ManifestStats',
    'NormalizationReport',
    'PartitionSummary',
    'PreparationReportSchema',
    'PreparedSchema',
    'ProvenanceRecord',
    'SchemaReport',
    'TargetHeadsSchema',
    'WorldGridReport',
    # constants
    'PREPARED_SCHEMA_ID',
]


# for static check
if typing.TYPE_CHECKING:
    from .grid import (
        GridReportSchema,
        WorldGridReport,
    )
    from .harmonization import (
        HarmonizationReportSchema,
        HarmonizationRunRecord,
        ProvenanceRecord,
    )
    from .ingestion import (
        BlockStats,
        DataBlocksReport,
        DomainMapReport,
        DomainStats,
        IngestReportSchema,
        ManifestStats,
    )
    from .preparation import (
        BlocksPartition,
        DataPartitionReport,
        ImageBandStats,
        NormalizationReport,
        PREPARED_SCHEMA_ID,
        PartitionSummary,
        PreparationReportSchema,
        PreparedSchema,
        SchemaReport,
        TargetHeadsSchema,
    )


def __getattr__(name: str):
    if name in {
        'GridReportSchema',
        'WorldGridReport',
    }:
        obj = importlib.import_module('.grid', __package__)
        return getattr(obj, name)

    if name in {
        'HarmonizationReportSchema',
        'HarmonizationRunRecord',
        'ProvenanceRecord',
    }:
        obj = importlib.import_module('.harmonization', __package__)
        return getattr(obj, name)

    if name in {
        'BlockStats',
        'DataBlocksReport',
        'DomainMapReport',
        'DomainStats',
        'IngestReportSchema',
        'ManifestStats',
    }:
        obj = importlib.import_module('.ingestion', __package__)
        return getattr(obj, name)

    if name in {
        'BlocksPartition',
        'DataPartitionReport',
        'ImageBandStats',
        'NormalizationReport',
        'PREPARED_SCHEMA_ID',
        'PartitionSummary',
        'PreparationReportSchema',
        'PreparedSchema',
        'SchemaReport',
        'TargetHeadsSchema',
    }:
        obj = importlib.import_module('.preparation', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
