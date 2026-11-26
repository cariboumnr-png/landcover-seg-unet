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
Top-level namespace for `landseg.testing`.

Provides synthetic GeoTIFF and geospatial dataset generators for
integration tests and local development via lazy resolution.

Public APIs:
    - `PreHarmonizationDummyPaths`: Pre-harmonization dummy paths.
    - `PreMosaickingDummyPaths`: Pre-mosaicking dummy paths.
    - `TIFFConfig`: Configuration for dummy GeoTIFF generation.
    - `create_dummy_geotiff`: Write a dummy GeoTIFF file from config.
    - `generate_pre_harmonization_data`: Build pre-harmonization data.
    - `generate_pre_mosaic_data`: Generate cross-UTM dummy mosaic data.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'PreHarmonizationDummyPaths',
    'PreMosaickingDummyPaths',
    'TIFFConfig',
    'create_dummy_geotiff',
    'generate_pre_harmonization_data',
    'generate_pre_mosaic_data',
]


# for static check
if typing.TYPE_CHECKING:
    from .pre_harmonize import (
        PreHarmonizationDummyPaths,
        generate_pre_harmonization_data,
    )
    from .pre_mosaic import (
        PreMosaickingDummyPaths,
        generate_pre_mosaic_data,
    )
    from .tiff import (
        TIFFConfig,
        create_dummy_geotiff,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.pre_harmonize': [
        'PreHarmonizationDummyPaths',
        'generate_pre_harmonization_data',
    ],
    '.pre_mosaic': [
        'PreMosaickingDummyPaths',
        'generate_pre_mosaic_data',
    ],
    '.tiff': [
        'TIFFConfig',
        'create_dummy_geotiff',
    ],
})
