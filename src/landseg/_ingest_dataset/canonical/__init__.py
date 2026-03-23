# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Top-level namespace for `landseg._ingest_dataset`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'AlignmentConfig',
    'BlockBuilder',
    'BlocksCatalog',
    'CatalogueInputs',
    'DataBlock',
    # functions
    'align_rasters',
    'build_catalogue',
    # typing
    'BuilderConfig',
]

# for static check
if typing.TYPE_CHECKING:
    from .align import AlignmentConfig, align_rasters
    from .blocks import BlockBuilder, BlocksCatalog, BuilderConfig, DataBlock
    from .pipeline import CatalogueInputs, build_catalogue

def __getattr__(name: str):

    if name in ['AlignmentConfig', 'align_rasters']:
        return getattr(importlib.import_module('.align', __package__), name)
    if name in ['BlockBuilder', 'BlocksCatalog', 'BuilderConfig', 'DataBlock']:
        return getattr(importlib.import_module('.blocks', __package__), name)
    if name in ['CatalogueInputs', 'build_catalogue']:
        return getattr(importlib.import_module('.pipeline', __package__), name)

    raise AttributeError(name)
