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

Exposes reporting schemas and specialized logger via lazy resolution.

Public APIs:
    - DataPartitionReport: report for dataset splitting and hydration.
    - NormalizationReport: report for block materialization and stats.
    - SchemaReport: report for dataset schema generation.
    - PreparationReportSchema: root summary schema for prepare pipeline.
    - PreparationLogger: logger collecting preparation execution reports.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'DataPartitionReport',
    'NormalizationReport',
    'SchemaReport',
    'PreparationReportSchema',
    'PreparationLogger',
]


# for static check
if typing.TYPE_CHECKING:
    from .schema import (
        DataPartitionReport,
        NormalizationReport,
        SchemaReport,
        PreparationReportSchema,
    )
    from .logger import PreparationLogger


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

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
