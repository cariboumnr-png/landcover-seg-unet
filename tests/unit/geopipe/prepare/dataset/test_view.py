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

'''Unit tests for dataset view orchestration (view.py).'''

# standard imports
import typing
# third-party imports
import pytest
import rasterio.transform
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.prepare.dataset.catalog as catalog
import landseg.geopipe.prepare.dataset.view as view_mod


# ----- `build_dataset_view` tests
def test_build_dataset_view(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Ingested catalog, schema metadata, and user parameters.
    When: Running build_dataset_view.
    Then: Orchestrate and return a unified DatasetView.
    '''
    dummy_view = catalog.DataBlocksView(
        valid_blocks={(0, 0): 'path/to/block_0_0.h5'},
        external_test_blocks=None,
        crs='EPSG:3161',
        transform=rasterio.transform.Affine.identity(),
        raw_class_counts={(0, 0): {'landcover': [10, 20, 30]}},
    )

    dummy_schema: geo_core.DatasetSchema = typing.cast(
        geo_core.DatasetSchema,
        {
            'tensor_shapes': {
                'image': {'H': 256, 'W': 256},
            },
            'io_conventions': {
                'image_band_map': {'blue': 0, 'green': 1, 'red': 2},
                'label_band_map': {'landcover': 0},
            },
            'labels': {
                'label_num_cls': {'landcover': 2},
                'label_class_names': {'landcover': ['c1', 'c2']},
                'label_ignore_cls': {'landcover': [255]},
            },
            'dataset': {
                'image_schemes': {},
                'label_schemes': {
                    'landcover': {
                        'binary': {
                            'reclass': {'1': [1, 2]},
                            'reclass_name': {'1': 'VEG'},
                        }
                    }
                },
            },
        },
    )

    # mock read_catalog
    monkeypatch.setattr(
        catalog,
        'read_catalog',
        lambda *args, **kwargs: dummy_view,
    )

    params = view_mod.DatasetViewParameters(
        valid_pxs={'landcover': 0.5},
        features=['red', 'blue'],
        targets={'landcover': 'binary'},
    )

    dview = view_mod.build_dataset_view(
        'fake/catalog.json',
        dummy_schema,
        parameters=params,
        canvas_crs='EPSG:3161',
        canvas_transform=rasterio.transform.Affine.identity(),
    )

    # verify dataset view structure
    assert isinstance(dview, view_mod.DatasetView)
    assert dview.features.names == ('red', 'blue')
    assert dview.features.indices == (2, 0)
    assert dview.targets.head_names == [
        'landcover',
        'landcover_group',
        'landcover_VEG',
    ]
    assert dview.targets.head_parent['landcover_VEG'] == 'landcover_group'
    assert dview.targets.head_parent_cls['landcover_VEG'] == 1

    # verify in-memory derived class counts for reclassified focal head
    assert dview.focal_head == 'landcover_group'
    # group 0 = 10 + 20 = 30
    assert dview.valid_class_counts == {(0, 0): [30]}
    assert dview.base_class_counts == {(0, 0): [30]}
    assert dview.valid_blocks == {(0, 0): 'path/to/block_0_0.h5'}
    assert dview.crs == 'EPSG:3161'


def test_build_dataset_view_explicit_focal_target(
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Explicit focal_target requested in dataset parameters.
    When: Running build_dataset_view.
    Then: Correctly resolve focal head and populate matching counts.
    '''
    dummy_view = catalog.DataBlocksView(
        valid_blocks={(0, 0): 'path/to/block_0_0.h5'},
        external_test_blocks=None,
        crs='EPSG:3161',
        transform=rasterio.transform.Affine.identity(),
        raw_class_counts={(0, 0): {'landcover': [10, 20, 30]}},
    )

    dummy_schema: geo_core.DatasetSchema = typing.cast(
        geo_core.DatasetSchema,
        {
            'tensor_shapes': {
                'image': {'H': 256, 'W': 256},
            },
            'io_conventions': {
                'image_band_map': {'blue': 0},
                'label_band_map': {'landcover': 0},
            },
            'labels': {
                'label_num_cls': {'landcover': 2},
                'label_class_names': {'landcover': ['c1', 'c2']},
                'label_ignore_cls': {'landcover': [255]},
            },
            'dataset': {
                'image_schemes': {},
                'label_schemes': {
                    'landcover': {
                        'binary': {
                            'reclass': {'1': [1, 2]},
                            'reclass_name': {'1': 'VEG'},
                        }
                    }
                },
            },
        },
    )

    monkeypatch.setattr(
        catalog,
        'read_catalog',
        lambda *args, **kwargs: dummy_view,
    )

    params = view_mod.DatasetViewParameters(
        valid_pxs={'landcover': 0.5},
        focal_target='landcover',
        targets={'landcover': 'binary'},
    )

    dview = view_mod.build_dataset_view(
        'fake/catalog.json',
        dummy_schema,
        parameters=params,
        canvas_crs='EPSG:3161',
        canvas_transform=rasterio.transform.Affine.identity(),
    )

    assert dview.focal_head == 'landcover'
    assert dview.valid_class_counts == {(0, 0): [10, 20, 30]}
    assert dview.base_class_counts == {(0, 0): [10, 20, 30]}
