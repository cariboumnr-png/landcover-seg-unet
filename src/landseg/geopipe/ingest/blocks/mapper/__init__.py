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
Top-level namespace for `landseg.geopipe.ingest.blocks.mapper`.

Exposes raster-to-grid mapping, geometry validation, and window caching
utilities via lazy module resolution.

Public APIs:
    - GeometrySummary: TypedDict of raster geometry metadata.
    - MappedRasterWindows: Dataclass container for read windows.
    - map_rasters: Maps input rasters to grid and builds windows.
    - map_rasters_to_grid: Maps rasters onto grid with caching.
    - validate_geometry: Ingests rasters and validates alignment.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'map_rasters_to_grid',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .lifecycle import map_rasters_to_grid


def __getattr__(name: str):

    if name in {'map_rasters_to_grid'}:
        return getattr(
            importlib.import_module('.lifecycle', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
