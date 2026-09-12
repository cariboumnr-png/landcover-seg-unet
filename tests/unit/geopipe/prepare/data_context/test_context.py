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

'''Unit tests for dataset context orchestration.'''

# third-party imports
import pytest
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.prepare.data_context.catalog as catalog
import landseg.geopipe.prepare.data_context.context as context


# ----- test helper classes
class _DummyCatalogConfig:
    '''Dummy configuration satisfying `_CatalogViewConfig`.'''
    valid_pxs = {'landcover': 0.5}
    test_catalog = None
    non_overlapping_test_grid = False


# ----- `build_dataset_context` tests
def test_build_dataset_context(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Ingested catalog, schema metadata, and user configs.
    When: Running build_dataset_context.
    Then: Orchestrate and return a unified DatasetContext.
    '''
    dummy_view = catalog.DataBlocksView(
        valid_blocks={(0, 0): 'path/to/block_0_0.h5'},
        external_test_blocks=None,
        crs='EPSG:3161',
        transform=rasterio.transform.Affine.identity(),
    )

    dummy_schema = {
        'io_conventions': {
            'image_band_map': {'blue': 0, 'green': 1, 'red': 2},
        },
        'labels': {
            'label_class_names': {'landcover': ['c1', 'c2']},
            'label_ignore_cls': {'landcover': [255]},
        },
        'dataset': {
            'schemes': {
                'landcover': {
                    'binary': {
                        'reclass': {'1': [1, 2]},
                        'reclass_name': {'1': 'VEG'},
                    }
                }
            }
        },
    }

    # mock read_catalog
    monkeypatch.setattr(
        catalog,
        'read_catalog',
        lambda catalog_fpath, data_schema, config: dummy_view,
    )

    # mock artifacts controller load
    class _MockSchemaCtrl:
        def fetch(self):
            return dummy_schema

    monkeypatch.setattr(
        artifacts.Controller,
        'load_json_or_fail',
        lambda fpath: _MockSchemaCtrl(),
    )

    ctx = context.build_dataset_context(
        catalog_fpath='fake/catalog.json',
        schema_fpath='fake/schema.json',
        catalog_config=_DummyCatalogConfig(),
        user_features_cfg=['red', 'blue'],
        user_targets_cfg={'landcover': 'binary'},
    )

    # verify dataset context structure
    assert isinstance(ctx, context.DatasetContext)
    assert ctx.catalog == dummy_view
    assert ctx.features.names == ('red', 'blue')
    assert ctx.features.indices == (2, 0)
    assert ctx.targets.head_names == (
        'landcover',
        'landcover_VEG',
        'landcover_group',
    )
    assert ctx.targets.head_parent['landcover_VEG'] == 'landcover_group'
    assert ctx.targets.head_parent_cls['landcover_VEG'] == 1
