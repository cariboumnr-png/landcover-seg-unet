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

'''Unit tests for data context semantics resolution.'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.data_context.semantics as semantics


# ----- `resolve_feature_channels` tests
def test_resolve_feature_channels_default():
    '''
    Given: An available band mapping.
    When: Resolving feature channels with no user config.
    Then: Return all available bands in sequential index order.
    '''
    band_map = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}
    res = semantics.resolve_feature_channels(band_map, None, None)
    assert res.names == ('blue', 'green', 'red', 'nir')
    assert res.indices == (0, 1, 2, 3)


def test_resolve_feature_channels_named_scheme():
    '''
    Given: Feature schemes and user selection by scheme name.
    When: Resolving feature channels.
    Then: Correctly resolve selected band names and indices.
    '''
    band_map = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3, 'dem': 4}
    schemes = {
        'sentinel2': {
            'rgb': ['blue', 'green', 'red'],
            'rgb_nir': ['blue', 'green', 'red', 'nir'],
        }
    }
    user_cfg = {'sentinel2': 'rgb_nir'}
    res = semantics.resolve_feature_channels(band_map, user_cfg, schemes)
    assert res.names == ('blue', 'green', 'red', 'nir')
    assert res.indices == (0, 1, 2, 3)


def test_resolve_feature_channels_explicit_list():
    '''
    Given: User providing an explicit band list.
    When: Resolving feature channels.
    Then: Return requested bands in specified order.
    '''
    band_map = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}
    res = semantics.resolve_feature_channels(
        band_map, ['red', 'blue'], None
    )
    assert res.names == ('red', 'blue')
    assert res.indices == (2, 0)


def test_resolve_feature_channels_errors():
    '''
    Given: Missing scheme or unknown band name.
    When: Resolving feature channels.
    Then: Raise KeyError.
    '''
    band_map = {'blue': 0, 'green': 1}
    with pytest.raises(KeyError, match='feature scheme'):
        semantics.resolve_feature_channels(
            band_map,
            {'sentinel2': 'missing'},
            {'sentinel2': {'rgb': ['blue']}},
        )

    with pytest.raises(KeyError, match='feature band'):
        semantics.resolve_feature_channels(
            band_map, ['unknown'], None
        )


# ----- `resolve_target_heads` tests
def test_resolve_target_heads_default():
    '''
    Given: Layer names without user target reclassification.
    When: Resolving target heads.
    Then: Return single base head per layer with empty parents.
    '''
    label_names = {'landcover': ['conifer', 'decid', 'water']}
    ignore_cls = {'landcover': [255]}
    ctx = semantics.resolve_target_heads(
        label_names_map=label_names,
        user_targets_cfg=None,
        label_schemes=None,
        label_ignore_cls=ignore_cls,
    )
    assert ctx.head_names == ('landcover',)
    assert ctx.head_parent == {'landcover': None}
    assert ctx.head_parent_cls == {'landcover': None}
    assert ctx.num_classes == {'landcover': 3}
    assert ctx.class_names == {'landcover': ['conifer', 'decid', 'water']}
    assert ctx.ignore_classes == {'landcover': [255]}
    assert ctx.target_reclass == {'landcover': None}


def test_resolve_target_heads_reclass_with_names():
    '''
    Given: Reclass scheme containing human-readable `reclass_name`.
    When: Resolving target heads.
    Then: Produce base, named child heads, and group head.
    '''
    label_names = {'landcover': ['conifer', 'decid', 'water']}
    ignore_cls = {'landcover': [255]}
    schemes: dict[str, dict[str, geo_core.LabelScheme]] = {
        'landcover': {
            'binary': {
                'reclass': {'1': [1, 2], '2': [3]},
                'reclass_name': {'1': 'VEG', '2': 'WAT'},
            }
        }
    }
    user_cfg = {'landcover': 'binary'}

    ctx = semantics.resolve_target_heads(
        label_names_map=label_names,
        user_targets_cfg=user_cfg,
        label_schemes=schemes,
        label_ignore_cls=ignore_cls,
    )

    expected_heads = (
        'landcover',
        'landcover_VEG',
        'landcover_WAT',
        'landcover_group',
    )
    assert ctx.head_names == expected_heads

    # verify parent hierarchy
    assert ctx.head_parent == {
        'landcover': None,
        'landcover_VEG': 'landcover_group',
        'landcover_WAT': 'landcover_group',
        'landcover_group': None,
    }
    assert ctx.head_parent_cls == {
        'landcover': None,
        'landcover_VEG': 1,
        'landcover_WAT': 2,
        'landcover_group': None,
    }

    # verify class counts and class names
    assert ctx.num_classes == {
        'landcover': 3,
        'landcover_VEG': 2,
        'landcover_WAT': 1,
        'landcover_group': 2,
    }
    assert ctx.class_names == {
        'landcover': ['conifer', 'decid', 'water'],
        'landcover_VEG': ['conifer', 'decid'],
        'landcover_WAT': ['water'],
        'landcover_group': ['VEG', 'WAT'],
    }
    assert ctx.ignore_classes == {
        'landcover': [255],
        'landcover_VEG': [255],
        'landcover_WAT': [255],
        'landcover_group': [255],
    }


def test_resolve_target_heads_reclass_fallback_names():
    '''
    Given: Reclass scheme without `reclass_name`.
    When: Resolving target heads.
    Then: Fall back to easily understood auto-generated names.
    '''
    label_names = {'landcover': ['c1', 'c2', 'c3']}
    inline_scheme = {
        'reclass': {'1': [1, 2], '2': [3]}
    }
    user_cfg = {'landcover': inline_scheme}

    ctx = semantics.resolve_target_heads(
        label_names_map=label_names,
        user_targets_cfg=user_cfg,  # type: ignore[arg-type]
    )

    assert ctx.head_names == (
        'landcover',
        'landcover_sub1',
        'landcover_sub2',
        'landcover_group',
    )
    assert ctx.class_names['landcover_group'] == ['group_1', 'group_2']
    assert ctx.head_parent['landcover_sub1'] == 'landcover_group'
    assert ctx.head_parent_cls['landcover_sub1'] == 1


def test_resolve_target_heads_bypass_keywords():
    '''
    Given: User targets config set to 'base', 'raw', or 'none'.
    When: Resolving target heads.
    Then: Treat as un-reclassified base layer.
    '''
    label_names = {'landcover': ['c1', 'c2']}
    for kw in ('base', 'raw', 'none'):
        ctx = semantics.resolve_target_heads(
            label_names_map=label_names,
            user_targets_cfg={'landcover': kw},
        )
        assert ctx.head_names == ('landcover',)
        assert ctx.target_reclass['landcover'] is None


def test_resolve_target_heads_validation_errors():
    '''
    Given: Invalid schema structure or unrecognized scheme name.
    When: Resolving target heads.
    Then: Raise appropriate ValueError or TypeError.
    '''
    label_names = {'landcover': ['c1']}
    with pytest.raises(ValueError, match='must contain a "reclass"'):
        semantics.resolve_target_heads(
            label_names_map=label_names,
            user_targets_cfg={
                'landcover': {'invalid': 123}  # type: ignore[arg-type]
            },
        )

    with pytest.raises(TypeError, match='Invalid target config type'):
        semantics.resolve_target_heads(
            label_names_map=label_names,
            user_targets_cfg={'landcover': 12345},  # type: ignore[arg-type]
        )
