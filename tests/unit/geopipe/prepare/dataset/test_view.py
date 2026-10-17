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

'''Unit tests for dataset normalizer and view builder (view.py).'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.dataset.normalizer as normalizer
import landseg.geopipe.prepare.dataset.view as view_mod


# ----- test helpers
def _make_dummy_schema() -> geo_core.DatasetSchema:
    '''Construct minimal DatasetSchema for dataset view tests.'''
    return {
        'schema_id': 'data_schema/v1.1',
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


def _make_dummy_catalog() -> dict[str, geo_core.DatasetBlockMeta]:
    '''Construct minimal catalog mapping for dataset view tests.'''
    meta_0_0: geo_core.DatasetBlockMeta = {
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
    }
    meta_0_256: geo_core.DatasetBlockMeta = {
        'block_name': 'row_000256_col_000000',
        'file_path': '/path/block_256_0.npz',
        'row_col': [256, 0],
        'valid_px_ratios': {'landcover': 0.4},
        'class_count': {'landcover': [50, 150, 250]},
        'schema_version': 'v1.0',
        'creation_time': '2026-09-18T00:00:00',
        'sha_256': 'def456hash',
        'aligned_grid': 'grid_1',
        'source_image': '/path/img.tif',
        'source_image_sha_256': 'img_hash',
    }
    return {
        'row_000000_col_000000': meta_0_0,
        'row_000256_col_000000': meta_0_256,
    }


# ----- `read_dataset_manifest` tests
def test_read_dataset_manifest_basic():
    '''
    Given: Valid catalog entries and schema.
    When: `read_dataset_manifest` is evaluated.
    Then: Correctly parse base coordinates and valid class counts.
    '''
    inputs = normalizer.DataBlocksManifestInputs(
        schema=_make_dummy_schema(),
        catalog=_make_dummy_catalog(),
    )
    res = normalizer.read_dataset_manifest(inputs, focal_head='landcover')

    assert len(res.valid_blocks) == 2
    assert (0, 0) in res.valid_blocks
    assert (0, 256) in res.valid_blocks
    assert res.valid_class_counts[(0, 0)] == [100, 200, 300]
    assert res.base_class_counts[(0, 0)] == [100, 200, 300]


def test_read_dataset_manifest_threshold_filtering():
    '''
    Given: Catalog entries with varying valid_px_ratios.
    When: Setting a head_pxs_thres requirement of 0.8.
    Then: Filter out entries below the threshold.
    '''
    inputs = normalizer.DataBlocksManifestInputs(
        schema=_make_dummy_schema(),
        catalog=_make_dummy_catalog(),
    )
    res = normalizer.read_dataset_manifest(
        inputs,
        head_pxs_thres={'landcover': 0.8},
        focal_head='landcover',
    )
    # only (0, 0) has valid_px_ratio = 1.0 >= 0.8
    assert len(res.valid_blocks) == 1
    assert (0, 0) in res.valid_blocks
    assert (0, 256) not in res.valid_blocks


# ----- `build_dataset_view` tests
def test_build_dataset_view_success():
    '''
    Given: DataBlocksManifestInputs and configuration parameters.
    When: Calling `build_dataset_view`.
    Then: Construct integrated DatasetView with resolved features/targets.
    '''
    inputs = normalizer.DataBlocksManifestInputs(
        schema=_make_dummy_schema(),
        catalog=_make_dummy_catalog(),
    )
    cfg = view_mod.DatasetViewConfig(
        focal_head='landcover',
        features=['blue', 'green'],
    )
    ds_view = view_mod.build_dataset_view(inputs, cfg)

    assert ds_view.focal_head == 'landcover'
    assert ds_view.features.names == ('blue', 'green')
    assert ds_view.features.indices == (0, 1)
    assert len(ds_view.manifest.valid_blocks) == 2
    assert 'landcover' in ds_view.targets.head_names
