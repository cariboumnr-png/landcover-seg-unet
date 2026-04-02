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
Top-level namespace for `landseg.geopipe.foundation.domain_maps`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DomainBuildingParameters',
    'MappedDomainTiles',
    # functions
    'build_domain',
    'load_domain',
    'map_domain_to_grid',
    'prepare_domain_maps',
    'save_domain',
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .builder import build_domain
    from .domain_io import load_domain, save_domain
    from .lifecycle import DomainBuildingParameters, prepare_domain_maps
    from .mapper import MappedDomainTiles, map_domain_to_grid

def __getattr__(name: str):

    if name in {'build_domain'}:
        return getattr(importlib.import_module('.builder', __package__), name)

    if name in {'load_domain', 'save_domain'}:
        return getattr(importlib.import_module('.domain_io', __package__), name)

    if name in {'MappedDomainTiles', 'map_domain_to_grid'}:
        return getattr(importlib.import_module('.mapper', __package__), name)

    if name in {'DomainBuildingParameters', 'prepare_domain_maps'}:
        return getattr(importlib.import_module('.lifecycle', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
