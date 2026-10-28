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

'''Unit tests for geopipe ingest domain handlers (handlers.py).'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid.handlers as grid_handlers
import landseg.geopipe.ingest.handlers as handlers
import landseg.geopipe.rasters as rasters


# ----- `validate_block_feature_specs` tests
def test_validate_block_feature_specs():
    '''
    Given: A valid band list and feature parameters.
    When: `validate_block_feature_specs` is called via handlers.
    Then: Validation completes without error.
    '''
    bands = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'dem']
    handlers.validate_block_feature_specs(
        bands,
        add_spectral=['ndvi', 'ndmi'],
        add_topo=['slope'],
    )


# ----- `validate_raster_alignment` tests
def test_validate_raster_alignment(dummy_geotiff_factory):
    '''
    Given: Opened image and label dataset readers matching a grid.
    When: `validate_raster_alignment` is called via handlers.
    Then: Return pixel size tuple and bounding box.
    '''
    ref_fp = dummy_geotiff_factory(
        filename='ref.tif', width=32, height=32, crs='EPSG:32617'
    )
    img_fp = dummy_geotiff_factory(
        filename='img.tif', width=32, height=32, crs='EPSG:32617'
    )
    grid = grid_handlers.validate_grid_inputs(
        tile_size=(16, 16),
        tile_stride=(8, 8),
        mode='ref',
        ref_fpath=pathlib.Path(ref_fp),
    )

    with rasters.read_rasters(pathlib.Path(img_fp), None) as (img, _):
        pixel_size, bbox = handlers.validate_raster_alignment(grid, img)

    assert len(pixel_size) == 2
    assert bbox.left < bbox.right


# ----- `validate_pool_grid_compatibility` tests
def test_validate_pool_grid_compatibility(tmp_path):
    '''
    Given: A target grid layout and matching or mismatching pool schema.
    When: `validate_pool_grid_compatibility` is called via handlers.
    Then: Pass on compatible schema and raise ArtifactError on mismatch.
    '''
    layout = _create_dummy_grid()
    schema_fp = tmp_path / 'schema.json'

    # non-existent schema file is valid (first run)
    handlers.validate_pool_grid_compatibility(schema_fp, layout)

    # matching schema file passes
    schema_data = {
        'schema_id': geo_core.data_schema.SCHEMA_ID,
        'dataset': {
            'name': 'test_pool',
            'last_updated': '2026-09-24T00:00:00Z',
            'dataprep_commit': 'dev',
            'mapped_grids': [layout.gid],
            'block_identity': layout.block_identity,
            'data_source': {'image_paths': [], 'label_paths': []},
            'image_schemes': {},
            'label_schemes': {},
        },
    }
    artifacts.Controller[dict](schema_fp).persist(schema_data)
    handlers.validate_pool_grid_compatibility(schema_fp, layout)

    # mismatching block identity raises ArtifactError
    mismatched_spec = geo_core.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(32, 32),
        tile_stride=(16, 16),
        grid_extent=(160.0, 160.0),
    )
    mismatched_layout = geo_core.GridLayout(mismatched_spec)
    with pytest.raises(
        artifacts.ArtifactError, match='Grid compatibility mismatch'
    ):
        handlers.validate_pool_grid_compatibility(schema_fp, mismatched_layout)


# ----- private helpers
def _create_dummy_grid() -> geo_core.GridLayout:
    '''Create a minimal in-memory grid layout.'''
    spec = geo_core.GridSpec(
        crs='EPSG:32617',
        origin=(0.0, 1000.0),
        pixel_size=(10.0, 10.0),
        tile_size=(16, 16),
        tile_stride=(8, 8),
        grid_extent=(80.0, 80.0),
    )
    return geo_core.GridLayout(spec)
