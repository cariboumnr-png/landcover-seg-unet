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

'''Unit tests for the canonical block-building pipeline module.'''

# standard imports
import dataclasses
import json
import os
# third-party imports
import numpy
import rasterio
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.grid.builder as grid_builder
import landseg.geopipe.ingest as ingest
import landseg.geopipe.ingest.blocks as blocks


@dataclasses.dataclass
class _Params:
    tile_size: tuple[int, int] = (256, 256)
    tile_stride: tuple[int, int] = (128, 128)
    ref_fpath: str | None = None
    crs_string: str | None = None
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None


# ----- pipeline execution
def test_pipeline_run_canonical_blocks(tmp_path, dummy_geotiff_factory):
    '''
    Given: A clean layout, grid parameters, and canonical raster inputs.
    When: Running data blocks building.
    Then: Compile canonical block datasets, manifest catalog,
        and schema.
    '''
    report_file = str(tmp_path / 'ingest_report.json')
    logger = ingest.IngestionLogger(
        name='test_ingest_canonical',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(
        run_id='test_run_canonical',
        timestamp='2026-07-08T18:00:00Z'
    )

    img = dummy_geotiff_factory(
        filename='comp_11band.tif',
        width=256,
        height=256,
        bands=11,
        crs='EPSG:3161',
        dtype=numpy.float32
    )
    lbl = dummy_geotiff_factory(
        filename='label.tif',
        width=256,
        height=256,
        bands=1,
        crs='EPSG:3161',
        dtype=numpy.uint8
    )
    with rasterio.open(lbl, 'r+') as dataset:
        dataset.set_band_description(1, 'land_cover')
        dataset.update_tags(1, num_cls=2, ignore_cls='[255]', index_base=1)

    # prepare world grid from the reference raster
    grid_config = _Params(
        ref_fpath=str(img),
        crs_string='EPSG:3161',
        tile_size=(256, 256),
        tile_stride=(128, 128)
    )
    world_grid = grid_builder.build_grid('ref', grid_config)

    # initialize pipeline path containers in temp output directory
    paths = artifacts.IngestionPaths(str(tmp_path))

    # set pipeline configurations
    config = blocks.BlockBuildingParameters(
        image_fpath=str(img),
        label_fpath=str(lbl),
        dem_pad=8,
        ignore_index=255,
    )

    # run the pipeline
    blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger
    )

    # verify outputs
    db_paths = paths.data_blocks
    assert os.path.exists(db_paths.catalog)
    assert os.path.exists(db_paths.schema)
    assert os.path.exists(db_paths.blocks)

    # read and inspect catalog
    with open(db_paths.catalog, 'r', encoding='UTF-8') as f:
        catalog_data = json.load(f)
    assert len(catalog_data) > 0
    first_block = next(iter(catalog_data.values()))
    assert 'block_name' in first_block
    assert 'file_path' in first_block

    # read and inspect schema
    with open(db_paths.schema, 'r', encoding='UTF-8') as f:
        schema_data = json.load(f)
    assert schema_data['dataset']['block_identity'] == (
        world_grid.block_identity
    )

    # read and inspect report
    assert logger.summary is not None
    assert 'data_blocks' in logger.summary
    assert logger.summary['data_blocks'] is not None
    assert logger.summary['data_blocks']['image_filepath'] == str(img)
    assert logger.summary['data_blocks']['label_filepath'] == str(lbl)


def test_pipeline_block_building_parameters_features():
    '''
    Given: Custom feature arguments for `BlockBuildingParameters`.
    When: Instantiating `BlockBuildingParameters`.
    Then: Hold configured add_topo and add_spectral values.
    '''
    params = blocks.BlockBuildingParameters(
        image_fpath='img.tif',
        label_fpath='lbl.tif',
        dem_pad=8,
        ignore_index=255,
        add_spectral=['ndvi', 'ndmi'],
        add_topo=True,
    )
    assert params.add_topo is True
    assert params.add_spectral == ['ndvi', 'ndmi']
