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

'''Unit tests for preparation context resolution (context.py).'''

# standard imports
import dataclasses
# third-party imports
import pytest
import rasterio.crs
import rasterio.transform
# local imports
import landseg.geopipe.prepare.context as prep_context


# ----- `PreparationContext` tests
def test_preparation_context_frozen():
    '''
    Given: Instantiated PreparationContext.
    When: Attempting to mutate an attribute.
    Then: Raise FrozenInstanceError.
    '''
    ctx = prep_context.PreparationContext(
        catalog_fpath='/path/catalog.json',
        schema_fpath='/path/schema.json',
        schema={},
        canvas_crs='EPSG:32617',
        canvas_transform=rasterio.transform.Affine.identity(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        # pylint: disable=dataclass-cannot-be-modified
        ctx.canvas_crs = 'EPSG:4326'


def test_build_preparation_context_success(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Valid schema JSON and rasterio image.
    When: Running `build_preparation_context`.
    Then: Correctly load schema and canvas spatial reference metadata.
    '''
    dummy_schema = {
        'dataset': {
            'data_source': {
                'image_paths': ['/path/to/image.tif'],
            }
        }
    }

    class _MockSchemaCtrl:
        def fetch(self):
            return dummy_schema

    monkeypatch.setattr(
        prep_context.DatasetSchemaCtrl,
        'load_json_or_fail',
        lambda fpath: _MockSchemaCtrl(),
    )

    class _MockRasterSrc:
        crs = rasterio.crs.CRS.from_epsg(32617)
        transform = rasterio.transform.Affine(10, 0, 100, 0, -10, 200)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(
        prep_context.rasterio,
        'open',
        lambda fpath: _MockRasterSrc(),
    )

    ctx = prep_context.build_preparation_context(
        catalog_fpath='/path/catalog.json',
        schema_fpath='/path/schema.json',
    )

    assert ctx.catalog_fpath == '/path/catalog.json'
    assert ctx.schema_fpath == '/path/schema.json'
    assert ctx.schema == dummy_schema
    assert ctx.canvas_crs == 'EPSG:32617'
    assert ctx.canvas_transform == _MockRasterSrc.transform


def test_build_preparation_context_no_crs(monkeypatch: pytest.MonkeyPatch):
    '''
    Given: Raster with missing CRS.
    When: Running `build_preparation_context`.
    Then: Raise ValueError.
    '''
    dummy_schema = {
        'dataset': {
            'data_source': {
                'image_paths': ['/path/to/image.tif'],
            }
        }
    }

    class _MockSchemaCtrl:
        def fetch(self):
            return dummy_schema

    monkeypatch.setattr(
        prep_context.DatasetSchemaCtrl,
        'load_json_or_fail',
        lambda fpath: _MockSchemaCtrl(),
    )

    class _MockNoCrsSrc:
        crs = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    monkeypatch.setattr(
        prep_context.rasterio,
        'open',
        lambda fpath: _MockNoCrsSrc(),
    )

    with pytest.raises(ValueError, match='Raster has no CRS'):
        prep_context.build_preparation_context(
            catalog_fpath='/path/catalog.json',
            schema_fpath='/path/schema.json',
        )
