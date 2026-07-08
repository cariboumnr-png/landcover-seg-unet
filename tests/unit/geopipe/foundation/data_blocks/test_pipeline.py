# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

# pylint: disable=missing-function-docstring

'''Unit tests for the canonical block-building pipeline module.'''

# standard imports
import json
import os
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.foundation as foundation
import landseg.geopipe.foundation.common as common
import landseg.geopipe.foundation.data_blocks.pipeline as pipeline

# Absolute path to the repo root folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../'))


# ----- fixtures
@pytest.fixture
def dummy_data_paths():
    '''Fixture providing paths to the pre-generated dummy data under experiment/input.'''
    ref_fpath = os.path.join(ROOT_DIR, 'experiment', 'input', 'extent_reference', 'example_extent.tif')
    dev_image = os.path.join(ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'dev', 'example_image.tif')
    dev_label = os.path.join(ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'dev', 'example_label.tif')
    test_image = os.path.join(ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'test', 'example_image.tif')
    test_label = os.path.join(ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'test', 'example_label.tif')
    dataset_config = os.path.join(ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'example_config.json')

    return {
        'ref_fpath': ref_fpath,
        'dev_image': dev_image,
        'dev_label': dev_label,
        'test_image': test_image,
        'test_label': test_label,
        'dataset_config': dataset_config
    }


# ----- pipeline execution
def test_pipeline_run_dev_stage(tmp_path, dummy_data_paths):
    # Setup logger and execution summary
    report_file = os.path.join(tmp_path, 'ingest_report.json')
    logger = common.FoundationLogger(
        name='test_ingest_dev',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary('test_run_dev', '2026-07-08T18:00:00Z')

    # Prepare world grid from the reference raster
    grid_config = foundation.GridParameters(
        mode='ref',
        crs='EPSG:2958',
        ref_fpath=dummy_data_paths['ref_fpath'],
        origin=(0.0, 0.0),
        pixel_size=(0.0, 0.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(256, 256, 128, 128)
    )
    grid_file = os.path.join(tmp_path, 'grid.json')
    world_grid = foundation.prepare_world_grid(
        grid_file,
        grid_config,
        policy=artifacts.LifecyclePolicy.REBUILD,
        logger=logger
    )

    # Initialize pipeline path containers in temp output directory
    paths = artifacts.FoundationPaths(str(tmp_path))

    # Set pipeline configurations
    config = pipeline.BlockBuildingParameters(
        stage='dev',
        image_fpath=dummy_data_paths['dev_image'],
        label_fpath=dummy_data_paths['dev_label'],
        data_config_fpath=dummy_data_paths['dataset_config'],
        dem_pad=8,
        ignore_index=255
    )

    # Run the pipeline
    pipeline.run_blocks_building(
        world_grid,
        paths.data_blocks.dev,
        config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger
    )

    # Verify outputs under dev partition
    dev_paths = paths.data_blocks.dev
    assert os.path.exists(dev_paths.catalog)
    assert os.path.exists(dev_paths.schema)
    assert os.path.exists(dev_paths.blocks)

    # Read and inspect catalog
    with open(dev_paths.catalog, 'r') as f:
        catalog_data = json.load(f)
    assert len(catalog_data) > 0
    first_key = list(catalog_data.keys())[0]
    assert 'block_name' in catalog_data[first_key]
    assert 'file_path' in catalog_data[first_key]

    # Verify logger records reports correctly
    assert logger.summary is not None
    assert 'data_blocks' in logger.summary
    assert 'dev' in logger.summary['data_blocks']
    report = logger.summary['data_blocks']['dev']
    assert report['image_filepath'] == dummy_data_paths['dev_image']
    assert report['label_filepath'] == dummy_data_paths['dev_label']


def test_pipeline_run_test_stage(tmp_path, dummy_data_paths):
    # Setup logger and execution summary
    report_file = os.path.join(tmp_path, 'ingest_report.json')
    logger = common.FoundationLogger(
        name='test_ingest_test',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary('test_run_test', '2026-07-08T18:00:00Z')

    # Prepare world grid from the reference raster
    grid_config = foundation.GridParameters(
        mode='ref',
        crs='EPSG:2958',
        ref_fpath=dummy_data_paths['ref_fpath'],
        origin=(0.0, 0.0),
        pixel_size=(0.0, 0.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(256, 256, 128, 128)
    )
    grid_file = os.path.join(tmp_path, 'grid.json')
    world_grid = foundation.prepare_world_grid(
        grid_file,
        grid_config,
        policy=artifacts.LifecyclePolicy.REBUILD,
        logger=logger
    )

    # Initialize pipeline path containers in temp output directory
    paths = artifacts.FoundationPaths(str(tmp_path))

    # Set pipeline configurations
    config = pipeline.BlockBuildingParameters(
        stage='test',
        image_fpath=dummy_data_paths['test_image'],
        label_fpath=dummy_data_paths['test_label'],
        data_config_fpath=dummy_data_paths['dataset_config'],
        dem_pad=8,
        ignore_index=255
    )

    # Run the pipeline
    pipeline.run_blocks_building(
        world_grid,
        paths.data_blocks.test,
        config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger
    )

    # Verify outputs under test partition
    test_paths = paths.data_blocks.test
    assert os.path.exists(test_paths.catalog)
    assert os.path.exists(test_paths.schema)
    assert os.path.exists(test_paths.blocks)

    # Read and inspect catalog
    with open(test_paths.catalog, 'r') as f:
        catalog_data = json.load(f)
    assert len(catalog_data) > 0
    first_key = list(catalog_data.keys())[0]
    assert 'block_name' in catalog_data[first_key]
    assert 'file_path' in catalog_data[first_key]

    # Verify logger records reports correctly
    assert logger.summary is not None
    assert 'data_blocks' in logger.summary
    assert 'test' in logger.summary['data_blocks']
    report = logger.summary['data_blocks']['test']
    assert report['image_filepath'] == dummy_data_paths['test_image']
    assert report['label_filepath'] == dummy_data_paths['test_label']
