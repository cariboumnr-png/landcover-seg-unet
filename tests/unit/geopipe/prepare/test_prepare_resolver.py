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

# pylint: disable=protected-access

'''Unit tests for feature channels and target reclassification resolver.'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.geopipe.prepare.normal_blocks.normalize as normalize
import landseg.geopipe.prepare.resolver as resolver


# ----- `resolve_feature_channels` tests
def test_resolve_feature_channels_default():
    '''
    Given: An available band mapping.
    When: Running resolve_feature_channels with no user config.
    Then: Return all available bands in sequential order.
    '''
    band_map = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'dem': 4}
    names, indices = resolver.resolve_feature_channels(
        band_map, None, None
    )
    assert names == ['blue', 'green', 'red', 'nir', 'dem']
    assert indices == [0, 1, 2, 3, 4]


def test_resolve_feature_channels_named_scheme():
    '''
    Given: Raster schemes metadata and user selecting a scheme name.
    When: Running resolve_feature_channels.
    Then: Correctly resolve selected band names and indices.
    '''
    band_map = {
        'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'swir1': 4, 'dem': 5
    }
    schemes = {
        'sentinel2': {
            'rgb': ['blue', 'green', 'red'],
            'rgb_nir': ['blue', 'green', 'red', 'nir'],
        }
    }
    user_cfg = {'sentinel2': 'rgb_nir', 'dem': 'all'}
    names, indices = resolver.resolve_feature_channels(
        band_map, user_cfg, schemes
    )
    assert names == ['blue', 'green', 'red', 'nir', 'dem']
    assert indices == [0, 1, 2, 3, 5]


def test_resolve_feature_channels_inline_list():
    '''
    Given: User providing an inline list of band names.
    When: Running resolve_feature_channels.
    Then: Return the specific selected bands and indices.
    '''
    band_map = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'dem': 4}
    user_cfg = {'custom': ['red', 'nir', 'dem']}
    names, indices = resolver.resolve_feature_channels(
        band_map, user_cfg, None
    )
    assert names == ['red', 'nir', 'dem']
    assert indices == [2, 3, 4]


def test_resolve_feature_channels_invalid():
    '''
    Given: Non-existent scheme or unknown band name.
    When: Running resolve_feature_channels.
    Then: Raise ValueError.
    '''
    band_map = {'blue': 0, 'green': 1}
    with pytest.raises(
        ValueError, match='Named feature scheme \"bad\" not found'
    ):
        resolver.resolve_feature_channels(
            band_map,
            {'sentinel2': 'bad'},
            {'sentinel2': {'rgb': ['blue']}},
        )

    with pytest.raises(ValueError, match='Band \"unknown\"'):
        resolver.resolve_feature_channels(
            band_map, {'custom': ['unknown']}, None
        )


# ----- `resolve_target_reclass` tests
def test_resolve_target_reclass():
    '''
    Given: User target configs (named schemes and inline dicts).
    When: Running resolve_target_reclass.
    Then: Correctly resolve target reclass configurations.
    '''
    label_names = {'landcover': ['coniferous', 'deciduous', 'water']}
    schemes = {
        'landcover': {
            'binary': {
                'reclass': {'1': [1, 2], '2': [3]},
                'reclass_name': {'1': 'VEG', '2': 'WAT'},
            }
        }
    }

    # default
    assert resolver.resolve_target_reclass(label_names, None, None) == {
        'landcover': None
    }

    # named scheme
    res = resolver.resolve_target_reclass(
        label_names, {'landcover': 'binary'}, schemes
    )
    assert res['landcover'] == schemes['landcover']['binary']

    # inline dict
    inline = {
        'reclass': {'1': [1]},
        'reclass_name': {'1': 'CONIFER'},
    }
    res_inline = resolver.resolve_target_reclass(
        label_names, {'landcover': inline}, None
    )
    assert res_inline['landcover'] == inline


# ----- `_reclassify_label_stack` tests
def test_reclassify_label_stack():
    '''
    Given: Raw 2D label array and active target reclassification.
    When: Running _reclassify_label_stack.
    Then: Construct base layer, child slices, and grouping layer.
    '''
    raw_arr = numpy.array([
        [1, 2],
        [3, 255]
    ], dtype=numpy.uint8)

    reclass_cfg = {
        'landcover': {
            'reclass': {'1': [1, 2], '2': [3]},
            'reclass_name': {'1': 'VEG', '2': 'WAT'},
        }
    }

    stack = normalize._reclassify_label_stack(
        [raw_arr], ['landcover'], reclass_cfg, ignore_index=255
    )
    # expected 4 layers: base (1..3), child 1 (1..2 reindexed to 1,2),
    # child 2 (3 reindexed to 1), group layer (1, 2)
    assert stack.shape == (4, 2, 2)
    # base layer
    assert numpy.array_equal(stack[0], raw_arr)
    # child 1 (classes 1 and 2)
    assert stack[1, 0, 0] == 1
    assert stack[1, 0, 1] == 2
    assert stack[1, 1, 0] == 255 # class 3 is masked
    # child 2 (class 3 reindexed to 1)
    assert stack[2, 0, 0] == 255
    assert stack[2, 1, 0] == 1


