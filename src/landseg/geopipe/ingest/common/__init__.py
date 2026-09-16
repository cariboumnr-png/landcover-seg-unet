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
Top-level namespace for `landseg.geopipe.ingest.common`.

Exposes common logging, report schemas, and aliases for data ingestion.

Public APIs:
    - `DomainStats`: TypedDict for domain layer re-indexing statistics.
    - `DomainMapReport`: TypedDict for domain map execution report.
    - `BlockStats`: TypedDict for data block mapping and build stats.
    - `ManifestStats`: TypedDict for catalog/schema update details.
    - `DataBlocksReport`: TypedDict for data block execution report.
    - `IngestReportSchema`: TypedDict for root data ingestion report.
    - `IngestionLogger`: Logger tracking ingest execution and report JSON.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    'DomainStats',
    'DomainMapReport',
    'BlockStats',
    'ManifestStats',
    'DataBlocksReport',
    'IngestReportSchema',
    'IngestionLogger',
]

# for static check
if typing.TYPE_CHECKING:
    from .schema import (
        DomainStats,
        DomainMapReport,
        BlockStats,
        ManifestStats,
        DataBlocksReport,
        IngestReportSchema,
    )
    from .logger import IngestionLogger


def __getattr__(name: str):
    if name in {
        'DomainStats',
        'DomainMapReport',
        'BlockStats',
        'ManifestStats',
        'DataBlocksReport',
        'IngestReportSchema',
    }:
        return getattr(importlib.import_module('.schema', __package__), name)

    if name in {'IngestionLogger'}:
        return getattr(importlib.import_module('.logger', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
