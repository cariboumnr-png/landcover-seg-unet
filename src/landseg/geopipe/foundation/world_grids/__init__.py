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
Top-level namespace for `landseg.geopipe.foundation.world_grids`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'GridExtentConfig',
    'GridGenerationConfig',
    # functions
    'load_grid',
    'prep_world_grid',
    'save_grid',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .api import GridExtentConfig, GridGenerationConfig, prep_world_grid
    from .io import load_grid, save_grid

def __getattr__(name: str):

    if name in ['GridExtentConfig', 'GridGenerationConfig', 'prep_world_grid']:
        return getattr(importlib.import_module('.api', __package__), name)
    if name in ['load_grid', 'save_grid']:
        return getattr(importlib.import_module('.io', __package__), name)

    raise AttributeError(name)
