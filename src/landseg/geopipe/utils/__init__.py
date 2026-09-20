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
Top-level namespace for `landseg.geopipe.utils`.

Exposes utility functions via lazy resolution to keep imports simple
and circular-free.

Public APIs:
    - `name_xy`: Convert block name string to (x, y) coordinate tuple.
    - `open_rasters`: Context manager yielding opened raster readers.
    - `xy_name`: Convert (x, y) coordinate tuple to block name string.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # functions
    'name_xy',
    'open_rasters',
    'xy_name',
]


# for static check
if typing.TYPE_CHECKING:
    from .coords_str import (
        name_xy,
        xy_name,
    )
    from .raster_context import (
        open_rasters,
    )


def __getattr__(name: str):
    if name in {
        'name_xy',
        'xy_name',
    }:
        obj = importlib.import_module('.coords_str', __package__)
        return getattr(obj, name)

    if name in {'open_rasters'}:
        obj = importlib.import_module('.raster_context', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
