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


# ----- test helpers
def _make_dummy_schema(
    *,
    image_bands: dict[str, int] | None = None,
    image_schemes: dict[str, dict[str, list[str]]] | None = None,
    label_bands: dict[str, int] | None = None,
    label_num_cls: dict[str, int] | None = None,
    label_class_names: dict[str, list[str]] | None = None,
    label_ignore_cls: dict[str, list[int]] | None = None,
    label_schemes: (
        dict[str, dict[str, geo_core.LabelScheme]] | None
    ) = None,
) -> geo_core.DataSchema:
    '''Construct mock DataSchema dictionary for testing.'''
    img_map = image_bands or {'blue': 0, 'green': 1, 'red': 2}
    lbl_map = label_bands or {'landcover': 0}
    num_cls = label_num_cls or {'landcover': 3}
    c_names = label_class_names or {
        'landcover': ['c1', 'c2', 'c3']
    }
    ign_cls = label_ignore_cls or {'landcover': [255]}
    return {
        'schema_id': 'data_schema/v1.1',
        'tensor_shapes': {'image': {'H': 256, 'W': 256}},
        'io_conventions': {
            'image_band_map': img_map,
            'label_band_map': lbl_map,
        },
        'labels': {
            'label_num_cls': num_cls,
            'label_class_names': c_names,
            'label_ignore_cls': ign_cls,
        },
        'dataset': {
            'image_schemes': image_schemes or {},
            'label_schemes': label_schemes or {},
        },
    }  # type: ignore[return-value]


# ----- `resolve_feature_channels` tests
def test_resolve_feature_channels_default():
    '''
    Given: An available band mapping.
    When: Resolving feature channels with no user config.
    Then: Return all available bands in sequential index order.
    '''
    schema = _make_dummy_schema(
        image_bands={'blue': 0, 'green': 1, 'red': 2, 'nir': 3}
    )
    res = semantics.resolve_feature_channels(schema, None)
    assert res.names == ('blue', 'green', 'red', 'nir')
    assert res.indices == (0, 1, 2, 3)


def test_resolve_feature_channels_named_scheme():
    '''
    Given: Feature schemes and user selection by scheme name.
    When: Resolving feature channels.
    Then: Correctly resolve selected band names and indices.
    '''
    schema = _make_dummy_schema(
        image_bands={
            'blue': 0,
            'green': 1,
            'red': 2,
            'nir': 3,
            'dem': 4,
        },
        image_schemes={
            'sentinel2': {
                'rgb': ['blue', 'green', 'red'],
                'rgb_nir': ['blue', 'green', 'red', 'nir'],
            }
        },
    )
    user_cfg = {'sentinel2': 'rgb_nir'}
    res = semantics.resolve_feature_channels(schema, user_cfg)
    assert res.names == ('blue', 'green', 'red', 'nir')
    assert res.indices == (0, 1, 2, 3)


def test_resolve_feature_channels_explicit_list():
    '''
    Given: User providing an explicit band list.
    When: Resolving feature channels.
    Then: Return requested bands in specified order.
    '''
    schema = _make_dummy_schema(
        image_bands={'blue': 0, 'green': 1, 'red': 2, 'nir': 3}
    )
    res = semantics.resolve_feature_channels(schema, ['red', 'blue'])
    assert res.names == ('red', 'blue')
    assert res.indices == (2, 0)


def test_resolve_feature_channels_errors():
    '''
    Given: Missing scheme or unknown band name.
    When: Resolving feature channels.
    Then: Raise KeyError.
    '''
    schema = _make_dummy_schema(
        image_bands={'blue': 0, 'green': 1},
        image_schemes={'sentinel2': {'rgb': ['blue']}},
    )
    with pytest.raises(KeyError, match='feature scheme'):
        semantics.resolve_feature_channels(
            schema, {'sentinel2': 'missing'}
        )

    with pytest.raises(KeyError, match='feature band'):
        semantics.resolve_feature_channels(schema, ['unknown'])


# ----- `resolve_target_heads` tests
def test_resolve_target_heads_default():
    '''
    Given: Layer names without user target reclassification.
    When: Resolving target heads.
    Then: Return single base head per layer with empty parents.
    '''
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 3},
        label_class_names={'landcover': ['conifer', 'decid', 'water']},
        label_ignore_cls={'landcover': [255]},
    )
    ctx = semantics.resolve_target_heads(schema, None)
    assert ctx.head_names == ['landcover']
    assert ctx.head_parent == {'landcover': None}
    assert ctx.head_parent_cls == {'landcover': None}
    assert ctx.num_classes == {'landcover': 3}
    assert ctx.class_names == {'landcover': ['conifer', 'decid', 'water']}
    assert ctx.ignore_classes == {'landcover': [255]}
    assert ctx.resolved_reclass == {'landcover': None}


def test_resolve_target_heads_reclass_with_names():
    '''
    Given: Reclass scheme containing human-readable `reclass_name`.
    When: Resolving target heads.
    Then: Produce base, group head, and named child heads.
    '''
    schemes: dict[str, dict[str, geo_core.LabelScheme]] = {
        'landcover': {
            'binary': {
                'reclass': {'1': [1, 2], '2': [3]},
                'reclass_name': {'1': 'VEG', '2': 'WAT'},
            }
        }
    }
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 3},
        label_class_names={'landcover': ['conifer', 'decid', 'water']},
        label_ignore_cls={'landcover': [255]},
        label_schemes=schemes,
    )
    user_cfg = {'landcover': 'binary'}
    ctx = semantics.resolve_target_heads(schema, user_cfg)

    expected_heads = [
        'landcover',
        'landcover_group',
        'landcover_VEG',
        'landcover_WAT',
    ]
    assert ctx.head_names == expected_heads

    # verify parent hierarchy
    assert ctx.head_parent == {
        'landcover': None,
        'landcover_group': None,
        'landcover_VEG': 'landcover_group',
        'landcover_WAT': 'landcover_group',
    }
    assert ctx.head_parent_cls == {
        'landcover': None,
        'landcover_group': None,
        'landcover_VEG': 1,
        'landcover_WAT': 2,
    }

    # verify class counts and class names
    assert ctx.num_classes == {
        'landcover': 3,
        'landcover_group': 2,
        'landcover_VEG': 2,
        'landcover_WAT': 1,
    }
    assert ctx.class_names == {
        'landcover': ['conifer', 'decid', 'water'],
        'landcover_group': ['VEG', 'WAT'],
        'landcover_VEG': ['conifer', 'decid'],
        'landcover_WAT': ['water'],
    }
    assert ctx.ignore_classes == {
        'landcover': [255],
        'landcover_group': [255],
        'landcover_VEG': [255],
        'landcover_WAT': [255],
    }
    assert ctx.resolved_reclass['landcover'] is not None


def test_resolve_target_heads_reclass_fallback_names():
    '''
    Given: Reclass scheme without `reclass_name`.
    When: Resolving target heads.
    Then: Fall back to easily understood auto-generated names.
    '''
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 3},
        label_class_names={'landcover': ['c1', 'c2', 'c3']},
    )
    inline_scheme = {
        'reclass': {'1': [1, 2], '2': [3]}
    }
    user_cfg = {'landcover': inline_scheme}

    ctx = semantics.resolve_target_heads(
        schema,
        user_cfg,  # type: ignore[arg-type]
    )

    assert ctx.head_names == [
        'landcover',
        'landcover_group',
        'landcover_sub1',
        'landcover_sub2',
    ]
    assert ctx.class_names['landcover_group'] == ['group_1', 'group_2']
    assert ctx.head_parent['landcover_sub1'] == 'landcover_group'
    assert ctx.head_parent_cls['landcover_sub1'] == 1


def test_resolve_target_heads_bypass_keywords():
    '''
    Given: User targets config set to 'base', 'raw', or 'none'.
    When: Resolving target heads.
    Then: Treat as un-reclassified base layer.
    '''
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 2},
        label_class_names={'landcover': ['c1', 'c2']},
    )
    for kw in ('base', 'raw', 'none'):
        ctx = semantics.resolve_target_heads(
            schema,
            {'landcover': kw},
        )
        assert ctx.head_names == ['landcover']
        assert ctx.resolved_reclass['landcover'] is None


def test_resolve_target_heads_validation_errors():
    '''
    Given: Invalid schema structure or unrecognized scheme name.
    When: Resolving target heads.
    Then: Raise appropriate ValueError or TypeError.
    '''
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 1},
        label_class_names={'landcover': ['c1']},
    )
    with pytest.raises(ValueError, match='must contain a "reclass"'):
        semantics.resolve_target_heads(
            schema,
            {'landcover': {'invalid': 123}},  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match='Invalid target config type'):
        semantics.resolve_target_heads(
            schema,
            {'landcover': 12345},  # type: ignore[arg-type]
        )


# ----- `FeatureSelection` unpack tests
def test_feature_selection_iteration():
    '''
    Given: A FeatureSelection instance.
    When: Unpacking as a 2-tuple.
    Then: Unpack into names and indices lists.
    '''
    sel = semantics.FeatureSelection(names=('a', 'b'), indices=(0, 1))
    names, indices = sel
    assert names == ['a', 'b']
    assert indices == [0, 1]


# ----- `derive_head_class_counts` tests
def test_derive_head_class_counts():
    '''
    Given: Raw base class counts and target heads hierarchy.
    When: Deriving head class counts in memory.
    Then: Calculate exact pixel counts for base, child, and group heads.
    '''
    reclass_scheme = {
        'reclass': {'1': [1, 2], '2': [3]},
        'reclass_name': {'1': 'VEG', '2': 'WAT'},
    }
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 3},
        label_class_names={'landcover': ['c1', 'c2', 'c3']},
    )
    heads_ctx = semantics.resolve_target_heads(
        schema,
        {'landcover': reclass_scheme},  # type: ignore[arg-type]
    )

    raw_counts = {'landcover': [50, 100, 200, 300]}
    derived = semantics.derive_head_class_counts(heads_ctx, raw_counts)

    # 1. base head
    assert derived['landcover'] == [50, 100, 200, 300]
    # 2. child slices
    # VEG: source classes 0 and 1 -> raw[0] = 50, raw[1] = 100
    assert derived['landcover_VEG'] == [50, 100]
    # WAT: source class 2 -> raw[2] = 200
    assert derived['landcover_WAT'] == [200]
    # 3. grouping head
    # group 0 = 50 + 100 = 150, group 1 = 200
    assert derived['landcover_group'] == [150, 200]


# ----- `resolve_focal_head` tests
def test_resolve_focal_head():
    '''
    Given: Target heads context and various focal target requests.
    When: Resolving the focal head name.
    Then: Correctly resolve head from head name, class name, or default.
    '''
    reclass_scheme = {
        'reclass': {'1': [1, 2], '2': [3]},
        'reclass_name': {'1': 'VEG', '2': 'WAT'},
    }
    schema = _make_dummy_schema(
        label_bands={'landcover': 0},
        label_num_cls={'landcover': 3},
        label_class_names={'landcover': ['c1', 'c2', 'c3']},
    )
    heads_ctx = semantics.resolve_target_heads(
        schema,
        {'landcover': reclass_scheme},  # type: ignore[arg-type]
    )

    # default: prefers grouping head if reclassified
    assert semantics.resolve_focal_head(heads_ctx, None) == (
        'landcover_group'
    )

    # direct head name match
    assert semantics.resolve_focal_head(
        heads_ctx, 'landcover_VEG'
    ) == 'landcover_VEG'
    assert semantics.resolve_focal_head(
        heads_ctx, 'landcover'
    ) == 'landcover'

    # class name match (VEG is class name in landcover_group)
    assert semantics.resolve_focal_head(
        heads_ctx, 'VEG'
    ) == 'landcover_group'

    # unknown target error
    with pytest.raises(KeyError, match='Focal target "unknown" not found'):
        semantics.resolve_focal_head(heads_ctx, 'unknown')
