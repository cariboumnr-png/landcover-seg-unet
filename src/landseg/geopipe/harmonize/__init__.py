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
Top-level namespace for `landseg.geopipe.harmonize`.

Exposes raster harmonization, manifest compilation, taxonomy resolution,
and logging APIs via lazy resolution to keep import order simple and
circular-free.

Public APIs:
    - `HarmonizationLogger`: Logger tracking ETL progress and report JSON.
    - `WorldGridReport`: TypedDict for world grid summary report.
    - `data_harmonization_pipeline`: pipeline runner.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'HarmonizationLogger',
    # functions
    'data_harmonization_pipeline',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .logger import HarmonizationLogger
    from .pipeline import data_harmonization_pipeline


def __getattr__(name: str):

    if name in {
        'HarmonizationLogger',
        'HarmonizationReportSchema',
        'WorldGridReport',
    }:
        return getattr(
            importlib.import_module('.common', __package__), name
        )

    if name in {
        'data_harmonization_pipeline',
    }:
        return getattr(
            importlib.import_module('.pipeline', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
