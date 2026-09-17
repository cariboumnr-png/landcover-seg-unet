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
Top-level namespace for `landseg.geopipe.grid`.

Exposes world grid construction and lifecycle management APIs via lazy
resolution to keep import order simple and circular-free.

Public APIs:
    - `GridParameters`: Protocol defining grid generation configuration.
    - `build_grid`: Construct a GridLayout from config or reference raster.
    - `prepare_world_grid`: Build or load a persisted world grid artifact.
    - `load_grid_from_config`: Load a world grid artifact from configuration.
    - `load_grid_from_fpath`: Load a world grid layout directly from file.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'load_grid_from_config',
    'load_grid_from_fpath',
    'prepare_world_grid',
]

# for static check
if typing.TYPE_CHECKING:
    from .lifecycle import (
        load_grid_from_config,
        load_grid_from_fpath,
        prepare_world_grid,
    )


def __getattr__(name: str):

    if name in {
        'load_grid_from_config',
        'load_grid_from_fpath',
        'prepare_world_grid',
    }:
        mod = importlib.import_module('.lifecycle', __package__)
        return getattr(mod, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
