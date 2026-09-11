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
Harmonization pipeline types.
'''

# standard imports
from __future__ import annotations
import typing


class CategoricalSpecs(typing.TypedDict):
    '''Typed dictionary for categorical raster specifications.'''
    # required
    index_base: int
    num_cls: int
    ignore_cls: list[int]
    # optional
    class_name: typing.NotRequired[dict[str, str]]
    color_map: typing.NotRequired[dict[str, list[int]]] # requires RGB
    taxonomy: typing.NotRequired[TaxonomySpecs]


class TaxonomySpecs(typing.TypedDict):
    '''Typed dictionary for domain taxonomy specification.'''
    profile: str
    canonical_indices: typing.NotRequired[dict[str, int]]


class LabelScheme(typing.TypedDict):
    '''Named reclassification scheme for label raster.'''
    reclass: dict[str, list[int]]
    reclass_name: dict[str, str]


LabelSchemes = dict[str, LabelScheme]
