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
Manifest schemas and type definitions for data harmonization.

This module defines structures and type aliases used to validate and
represent dataset raster entries and their category/scheme mappings.

Public APIs:
    - `AllowedCategory`: Type alias for valid raster categories.
    - `FeatureSchemes`: Type alias for feature band scheme mappings.
    - `LabelScheme`: Re-exported TypedDict for label reclassification scheme.
    - `LabelSchemes`: Re-exported alias for label reclassification schemes.
    - `ManifestEntry`: TypedDict defining per-raster configuration shape.
'''

# standard imports
from __future__ import annotations
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


LabelSchemes: typing.TypeAlias = dict[str, geo_core.LabelScheme]


FeatureSchemes: typing.TypeAlias = dict[str, list[str]]


# ----- public types
class ManifestEntry(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: str
    band_mapping: dict[int, str]
    category: AllowedCategory
    categorical_specs: geo_core.CategoricalSpec | None
    schemes: LabelSchemes | FeatureSchemes | None
