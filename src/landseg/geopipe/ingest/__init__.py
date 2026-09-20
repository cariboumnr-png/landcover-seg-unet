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
Top-level namespace for `landseg.geopipe.ingest`.

Coordinates the ingestion of harmonized geospatial rasters into
tiled domain maps and canonical data blocks, providing logging,
context resolution, and pipeline execution tools via lazy module
resolution.

Public APIs:
    - `IngestionLogger`: Structured logger for ingestion stages.
    - `run_data_ingestion`: pipeline runner.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'IngestionLogger',
    # functions
    'run_data_ingestion'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from .logger import IngestionLogger
    from .pipeline import run_data_ingestion


def __getattr__(name: str):
    if name in {'IngestionLogger'}:
        return getattr(
            importlib.import_module('.logger', __package__), name
        )

    if name in {'run_data_ingestion'}:
        return getattr(
            importlib.import_module('.pipeline', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
