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

'''Unit tests for categorical raster specification types.'''

# local imports
import landseg.geopipe.core as geo_core


# ----- `CategoricalSpec` tests
def test_categorical_spec_structure():
    '''
    Given: Mandatory categorical spec attributes.
    When: Instantiating a CategoricalSpec dictionary.
    Then: All fields conform to specification contracts.
    '''
    spec: geo_core.CategoricalSpec = {
        'index_base': 1,
        'num_cls': 3,
        'ignore_cls': [255],
    }
    assert spec['index_base'] == 1
    assert spec['num_cls'] == 3
    assert spec['ignore_cls'] == [255]


def test_taxonomy_spec_and_label_scheme():
    '''
    Given: A TaxonomySpec and LabelScheme dictionary.
    When: Instantiating nested structures.
    Then: Successfully populates profile, indices, and reclass mappings.
    '''
    tax_spec: geo_core.TaxonomySpec = {
        'profile': 'ontario_landcover',
        'canonical_indices': {'1': 10, '2': 20},
    }
    cat_spec: geo_core.CategoricalSpec = {
        'index_base': 1,
        'num_cls': 2,
        'ignore_cls': [],
        'taxonomy': tax_spec,
    }
    scheme: geo_core.LabelScheme = {
        'reclass': {'1': [1, 2]},
        'reclass_name': {'1': 'vegetation'},
    }
    schemes: dict[str, geo_core.LabelScheme] = {'binary': scheme}

    assert cat_spec['taxonomy']['profile'] == 'ontario_landcover'
    assert 'binary' in schemes
    assert schemes['binary']['reclass_name']['1'] == 'vegetation'
