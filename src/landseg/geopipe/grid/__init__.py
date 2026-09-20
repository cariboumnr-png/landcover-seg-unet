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
    - `GridLogger`: Logger tracking world grid execution and report JSON.
    - `GridParameters`: Protocol defining grid generation config.
    - `build_grid`: Construct GridLayout from config or reference raster.
    - `get_grid_report_fpath`: Return canonical grid report file path.
    - `prepare_world_grid`: Build or load persisted world grid artifact.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'GridLogger',
    # functions
    'get_grid_report_fpath',
    'prepare_world_grid',
]


# for static check
if typing.TYPE_CHECKING:
    from .lifecycle import (
        get_grid_report_fpath,
        prepare_world_grid,
    )
    from .logger import (
        GridLogger,
    )


def __getattr__(name: str):
    if name in {
        'get_grid_report_fpath',
        'prepare_world_grid',
    }:
        obj = importlib.import_module('.lifecycle', __package__)
        return getattr(obj, name)

    if name in {'GridLogger'}:
        obj = importlib.import_module('.logger', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
