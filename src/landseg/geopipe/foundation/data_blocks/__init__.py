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
Top-level namespace for `landseg.geopipe.foundation.data_blocks`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockBuilder',
    'BlockBuilderConfig',
    'BlockBuildingParameters',
    'ManifestUpdateContext',
    'MappedRasterWindows',
    # functions
    'map_rasters_to_grid',
    'run_blocks_building',
    'update_manifest',
    # typing
    'PipelinePaths',
]

# for static check
if typing.TYPE_CHECKING:
    from .builder import BlockBuilder, BlockBuilderConfig
    from .common import PipelinePaths
    from .manifest import ManifestUpdateContext, update_manifest
    from .mapper import MappedRasterWindows, map_rasters_to_grid
    from .pipeline import BlockBuildingParameters, run_blocks_building

def __getattr__(name: str):

    if name in {'BlockBuilder', 'BlockBuilderConfig'}:
        return getattr(importlib.import_module('.builder', __package__), name)

    if name in {'PipelinePaths'}:
        return getattr(importlib.import_module('.common', __package__), name)

    if name in {'ManifestUpdateContext', 'update_manifest'}:
        return getattr(importlib.import_module('.manifest', __package__), name)

    if name in {'MappedRasterWindows', 'map_rasters_to_grid'}:
        return getattr(importlib.import_module('.mapper', __package__), name)

    if name in {'BlockBuildingParameters', 'run_blocks_building'}:
        return getattr(importlib.import_module('.pipeline', __package__), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
