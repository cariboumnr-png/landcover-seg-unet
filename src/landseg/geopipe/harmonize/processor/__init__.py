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
Top-level namespace for `landseg.geopipe.harmonize.processor`.

Exposes raster harmonization orchestration onto the canonical world grid.

Public APIs:
    - `HarmonizedRasters`: Container for processed raster paths dictionaries.
    - `harmonizer`: Harmonize compiled raster sources onto grid.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'HarmonizedRasters',
    'harmonizer',
]


# for static check
if typing.TYPE_CHECKING:
    from .processor import (
        HarmonizedRasters,
        harmonizer,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.processor': [
        'HarmonizedRasters',
        'harmonizer',
    ],
})
