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
Top-level namespace for `landseg.geopipe.harmonize.common`.

Exposes common logging and reporting utilities for data harmonization.

Public APIs:
    - `ProvenanceRecord`: TypedDict for raw raster file provenance.
    - `WorldGridReport`: TypedDict for world grid summary report.
    - `HarmonizationReportSchema`: TypedDict for overall pipeline report.
    - `HarmonizationLogger`: Logger tracking ETL progress and report JSON.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'ProvenanceRecord',
    'WorldGridReport',
    'HarmonizationReportSchema',
    'HarmonizationLogger',
]

# for static check
if typing.TYPE_CHECKING:
    from .schema import (
        ProvenanceRecord,
        WorldGridReport,
        HarmonizationReportSchema,
    )
    from .logger import HarmonizationLogger


def __getattr__(name: str):
    if name in {
        'ProvenanceRecord',
        'WorldGridReport',
        'HarmonizationReportSchema',
    }:
        return getattr(importlib.import_module('.schema', __package__), name)

    if name in {'HarmonizationLogger'}:
        return getattr(importlib.import_module('.logger', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
