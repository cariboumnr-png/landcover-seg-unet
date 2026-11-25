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
Top-level namespace for `landseg.geopipe.ingest.blocks.mapper`.

Exposes raster-to-grid mapping and window caching utilities via lazy
module resolution.

Public APIs:
    - `MappedRasterWindows`: Dataclass container for read windows.
    - `map_rasters_to_grid`: Maps rasters onto grid with caching.
    - `validate_raster_alignment`: Validates raster alignment with grid.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'MappedRasterWindows',
    'map_rasters_to_grid',
    'validate_raster_alignment',
]


# for static check
if typing.TYPE_CHECKING:
    from .lifecycle import map_rasters_to_grid
    from .mapper import (
        MappedRasterWindows,
        validate_raster_alignment,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.lifecycle': ['map_rasters_to_grid'],
    '.mapper': [
        'MappedRasterWindows',
        'validate_raster_alignment',
    ],
})
