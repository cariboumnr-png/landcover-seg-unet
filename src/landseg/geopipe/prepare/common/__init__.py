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
Top-level namespace for `landseg.geopipe.prepare.common`.

Exposes reporting schemas, artifact schemas, and specialized logger via
lazy resolution.

Public APIs:
    - DataPartitionReport: report for dataset splitting and hydration.
    - NormalizationReport: report for block materialization and stats.
    - SchemaReport: report for dataset schema generation.
    - PreparationReportSchema: root summary schema for prepare pipeline.
    - PreparationLogger: logger collecting preparation execution reports.
    - BlocksPartition: TypedDict mapping block IDs across splits.
    - ImageBandStats: TypedDict for image band statistics.
    - TargetHeadsSchema: TypedDict for target heads hierarchy.
    - PreparedSchema: TypedDict for dataset preparation schema.
    - PartitionSummary: TypedDict for split and hydration summary.
    - PREPARED_SCHEMA_ID: constant string for prepared schema ID.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # reports
    'DataPartitionReport',
    'NormalizationReport',
    'SchemaReport',
    'PreparationReportSchema',
    'PreparationLogger',
    # artifacts
    'BlocksPartition',
    'ImageBandStats',
    'TargetHeadsSchema',
    'PreparedSchema',
    'PartitionSummary',
    'PREPARED_SCHEMA_ID',
]


# for static check
if typing.TYPE_CHECKING:
    from .artifacts import (
        BlocksPartition,
        ImageBandStats,
        TargetHeadsSchema,
        PreparedSchema,
        PartitionSummary,
        PREPARED_SCHEMA_ID,
    )
    from .logger import PreparationLogger
    from .schema import (
        DataPartitionReport,
        NormalizationReport,
        SchemaReport,
        PreparationReportSchema,
    )


def __getattr__(name: str):
    if name in {
        'DataPartitionReport',
        'NormalizationReport',
        'SchemaReport',
        'PreparationReportSchema',
    }:
        return getattr(importlib.import_module('.schema', __package__), name)

    if name in {'PreparationLogger'}:
        return getattr(importlib.import_module('.logger', __package__), name)

    if name in {
        'BlocksPartition',
        'ImageBandStats',
        'TargetHeadsSchema',
        'PreparedSchema',
        'PartitionSummary',
        'PREPARED_SCHEMA_ID',
    }:
        return getattr(importlib.import_module('.artifacts', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
