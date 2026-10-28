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

'''Unit tests for geopipe prepare domain handlers (handlers.py).'''

# third-party imports
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.dataset as prepare_dataset
import landseg.geopipe.prepare.handlers as handlers


# ----- `validate_feature_channels` tests
def test_validate_feature_channels():
    '''
    Given: Ingested data schema with band mappings.
    When: `validate_feature_channels` is evaluated via handlers.
    Then: Return resolved feature channels or raise KeyError.
    '''
    schema = _make_dummy_schema()

    # valid default selection
    features = handlers.validate_feature_channels(schema, None)
    assert features.names == ('blue', 'green', 'red', 'nir')
    assert features.indices == (0, 1, 2, 3)

    # valid named scheme selection
    features = handlers.validate_feature_channels(schema, {'rgb': 'channels'})
    assert features.names == ('blue', 'green', 'red')

    # invalid band selection raises KeyError
    with pytest.raises(KeyError, match='Unknown feature band'):
        handlers.validate_feature_channels(schema, ['unknown_band'])


# ----- `validate_target_heads` tests
def test_validate_target_heads():
    '''
    Given: Ingested data schema with label metadata.
    When: `validate_target_heads` is evaluated via handlers.
    Then: Return resolved TargetHeadsContext.
    '''
    schema = _make_dummy_schema()
    targets = handlers.validate_target_heads(schema, None)
    assert 'landcover' in targets.head_names
    assert targets.num_classes['landcover'] == 3


# ----- `validate_focal_head` tests
def test_validate_focal_head():
    '''
    Given: Resolved TargetHeadsContext.
    When: `validate_focal_head` is evaluated via handlers.
    Then: Return matching focal head or raise KeyError on mismatch.
    '''
    schema = _make_dummy_schema()
    targets = handlers.validate_target_heads(schema, None)

    # valid focal target
    focal = handlers.validate_focal_head(targets, 'landcover')
    assert focal == 'landcover'

    # invalid focal target raises KeyError
    with pytest.raises(KeyError, match='not found'):
        handlers.validate_focal_head(targets, 'unknown_head')


# ----- `validate_dataset_view` tests
def test_validate_dataset_view():
    '''
    Given: Manifest inputs and view configuration.
    When: `validate_dataset_view` is evaluated via handlers.
    Then: Return compiled DatasetView.
    '''
    inputs = prepare_dataset.DataBlocksManifestInputs(
        schema=_make_dummy_schema(),
        catalog=_make_dummy_catalog(),
    )
    config = prepare_dataset.DatasetViewConfig(
        focal_head='landcover',
        features=['blue', 'green'],
    )
    view = handlers.validate_dataset_view(inputs, config)

    assert view.focal_head == 'landcover'
    assert view.features.names == ('blue', 'green')
    assert len(view.manifest.valid_blocks) == 2


# ----- private helpers
def _make_dummy_schema() -> geo_core.DataSchema:
    '''Construct minimal DataSchema dictionary for testing.'''
    return {
        'schema_id': geo_core.data_schema.SCHEMA_ID,
        'tensor_shapes': {'image': {'H': 256, 'W': 256}},
        'io_conventions': {
            'image_band_map': {'blue': 0, 'green': 1, 'red': 2, 'nir': 3},
            'label_band_map': {'landcover': 0},
        },
        'labels': {
            'label_num_cls': {'landcover': 3},
            'label_class_names': {'landcover': ['c1', 'c2', 'c3']},
            'label_ignore_cls': {'landcover': [255]},
        },
        'dataset': {
            'image_schemes': {'rgb': {'channels': ['blue', 'green', 'red']}},
            'label_schemes': {},
        },
    }  # type: ignore[return-value]


def _make_dummy_catalog() -> dict[str, geo_core.DataCatalogEntry]:
    '''Construct minimal catalog mapping for handler tests.'''
    meta_0: geo_core.DataCatalogEntry = {
        'block_name': 'row_000000_col_000000',
        'file_path': '/path/block_0_0.npz',
        'row_col': [0, 0],
        'valid_px_ratios': {'landcover': 1.0},
        'class_count': {'landcover': [100, 200, 300]},
        'schema_version': 'v1.0',
        'creation_time': '2026-09-18T00:00:00',
        'sha_256': 'abc123hash',
        'aligned_grid': 'grid_1',
        'source_image': '/path/img.tif',
        'source_image_sha_256': 'img_hash',
        'source_label': None,
        'source_label_sha_256': None,
    }
    meta_1: geo_core.DataCatalogEntry = {
        'block_name': 'row_000256_col_000000',
        'file_path': '/path/block_256_0.npz',
        'row_col': [256, 0],
        'valid_px_ratios': {'landcover': 0.9},
        'class_count': {'landcover': [50, 150, 250]},
        'schema_version': 'v1.0',
        'creation_time': '2026-09-18T00:00:00',
        'sha_256': 'def456hash',
        'aligned_grid': 'grid_1',
        'source_image': '/path/img.tif',
        'source_image_sha_256': 'img_hash',
        'source_label': None,
        'source_label_sha_256': None,
    }
    return {
        'row_000000_col_000000': meta_0,
        'row_000256_col_000000': meta_1,
    }
