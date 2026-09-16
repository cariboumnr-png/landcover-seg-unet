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
TypedDict report schemas for dataset preparation pipelines.

Defines schemas for data partitioning, block normalization, and schema
generation execution summaries collected by PreparationLogger.

Public APIs:
    - DataPartitionReport: report for dataset splitting and hydration.
    - NormalizationReport: report for block materialization and stats.
    - SchemaReport: report for dataset schema generation.
    - PreparationReportSchema: root summary schema for prepare pipeline.
'''

# standard imports
from __future__ import annotations
import typing


# ----- public types
class DataPartitionReport(typing.TypedDict):
    '''Execution report for dataset splitting and hydration.'''
    status: typing.Literal['loaded', 'created']
    duration_sec: float


class NormalizationReport(typing.TypedDict):
    '''Execution report for block normalization and materialization.'''
    status: typing.Literal['loaded', 'created']
    duration_sec: float
    unwanted_blocks_removed: int
    rebuild: bool
    stats_filepath: str


class SchemaReport(typing.TypedDict):
    '''Execution report for dataset schema generation.'''
    status: typing.Literal['loaded', 'created']
    duration_sec: float
    schema_filepath: str
    classes_mapped: list[str]


class PreparationReportSchema(typing.TypedDict):
    '''Root report mapping the entire data-prepare pipeline run.'''
    run_id: str
    timestamp: str
    status: typing.Literal['SUCCESS', 'FAILED']
    data_partition: DataPartitionReport | None
    normalization: NormalizationReport | None
    schema: SchemaReport | None
