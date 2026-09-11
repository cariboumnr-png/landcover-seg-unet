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
Data harmonization pipeline command implementation.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.geopipe.core as geo_core

# ----- public types
class ManifestEntry(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: str
    band_mapping: dict[int, str]
    category: AllowedCategory
    categorical_specs: geo_core.CategoricalSpecs | None
    schemes: LabelSchemes | FeatureSchemes | None


AllowedCategory = typing.Literal[
    'domains',
    'domain',
    'features',
    'feature',
    'labels',
    'label',
]


LabelScheme = geo_core.LabelScheme
LabelSchemes = geo_core.LabelSchemes
FeatureSchemes = dict[str, list[str]]
