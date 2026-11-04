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
Schemas and type contracts for dataset manifests.

Defines canonical TypedDict specifications, category aliases, and standard
sensor preset enumerations for raw raster registration.

Public APIs:
    - `AllowedCategory`: Type alias for valid raster categories.
    - `ManifestEntry`: TypedDict defining per-raster configuration.
    - `SensorPreset`: Canonical sensor band configurations and schemes.
'''

# standard imports
import enum
import pathlib
import typing
# local imports
import landseg.geopipe.core as geo_core


# ----- typing aliases
AllowedCategory: typing.TypeAlias = typing.Literal[
    'domains',
    'domain',
    'features',
    'feature',
    'labels',
    'label',
]


# ----- public types
class ManifestEntry(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: pathlib.Path
    band_mapping: dict[int, str]
    category: AllowedCategory
    categorical_specs: geo_core.CategoricalSpec | None
    schemes: dict[str, geo_core.LabelScheme] | dict[str, tuple[str, ...]] | None


# ----- public classes
class SensorPreset(enum.Enum):
    '''Standard remote sensing and geospatial raster band presets.'''

    # Sentinel-2 MSI L2A (10 surface reflectance bands)
    SENTINEL_2 = (
        'blue',
        'green',
        'red',
        'red_edge1',
        'red_edge2',
        'red_edge3',
        'nir',
        'narrow_nir',
        'swir1',
        'swir2',
    )

    # Landsat 8/9 OLI Surface Reflectance (7 optical bands)
    LANDSAT_8_9 = (
        'coastal',
        'blue',
        'green',
        'red',
        'nir',
        'swir1',
        'swir2',
    )

    # Standard 3-band true color
    RGB = (
        'red',
        'green',
        'blue',
    )

    # Standard 4-band color infrared
    RGB_NIR = (
        'red',
        'green',
        'blue',
        'nir',
    )

    # Topographic products (elevation + terrain derivatives)
    DEM_TERRAIN = (
        'elevation',
        'slope',
        'aspect',
    )
