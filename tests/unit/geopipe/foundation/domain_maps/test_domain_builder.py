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

# pylint: disable=protected-access

'''Unit tests for domain map builder (builder.py).'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.foundation.domain_maps.builder as domain_builder


# ----- `build_domain` tests
def test_build_domain_success():
    '''
    Given: A dict of mapped grid tiles including the max_idx marker.
    When: `build_domain` is executed.
    Then: Construct the DomainTileMap with resolved majorities,
        frequencies, and projected PCA features.
    '''
    # 2 classes: 0 and 1
    mapped_tiles = {
        (-999, -999): numpy.array([[1]]), # max_idx marker
        (0, 0): numpy.array([[0, 0], [0, 1]]), # 3 zeros, 1 one -> majority 0, major_freq 0.75
        (0, 1): numpy.array([[1, 1], [1, 1]]), # 4 ones -> majority 1, major_freq 1.0
        (1, 0): numpy.array([[-1, -1], [-1, -1]]), # all nodata -> filtered by valid_threshold
    }

    domain_map = domain_builder.build_domain(
        grid_id='grid_test',
        mapped_tiles=mapped_tiles,
        valid_threshold=0.5,
        target_variance=0.9
    )

    # valid tiles are (0,0) and (0,1). (1,0) should be excluded
    assert len(domain_map) == 2
    assert (0, 0) in domain_map
    assert (0, 1) in domain_map
    assert (1, 0) not in domain_map

    # verify majority classification
    assert domain_map[(0, 0)]['majority'] == 0
    assert domain_map[(0, 0)]['major_freq'] == 0.75
    assert domain_map[(0, 1)]['majority'] == 1
    assert domain_map[(0, 1)]['major_freq'] == 1.0

    # verify PCA features are present
    assert domain_map[(0, 0)]['pca_feature'] is not None
    assert len(domain_map[(0, 0)]['pca_feature']) > 0

    # verify metadata
    assert domain_map.meta['pca_axes_n'] > 0
    assert domain_map.meta['world_grid_ids'] == ['grid_test']


def test_k_from_target_evr_errors():
    '''
    Given: Invalid target variance values.
    When: Resolving PCA dimensions count.
    Then: Raise ValueError.
    '''
    evr = numpy.array([0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match='target must be in'):
        domain_builder._k_from_target_evr(evr, -0.1)

    with pytest.raises(ValueError, match='target must be in'):
        domain_builder._k_from_target_evr(evr, 1.5)
