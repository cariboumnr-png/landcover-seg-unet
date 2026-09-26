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

# pylint: disable=missing-function-docstring

'''Unit tests for intra-pool block collision policies and provenance.'''

# standard imports
import dataclasses
import json
import os
# third-party imports
import numpy
import pytest
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


# ----- fixtures
@pytest.fixture
def test_setup_factory(tmp_path, dummy_geotiff_factory):
    '''Prepare synthetic rasters and grid for collision tests.'''
    def _create():
        img = dummy_geotiff_factory(
            filename='comp_11band.tif',
            width=256,
            height=256,
            bands=11,
            crs='EPSG:3161',
            dtype=numpy.float32,
        )
        lbl = dummy_geotiff_factory(
            filename='label.tif',
            width=256,
            height=256,
            bands=1,
            crs='EPSG:3161',
            dtype=numpy.uint8,
        )
        with rasterio.open(lbl, 'r+') as dataset:
            dataset.set_band_description(1, 'land_cover')
            dataset.update_tags(
                1, num_cls=2, ignore_cls='[255]', index_base=1
            )

        grid_config = _Params(
            ref_fpath=str(img),
            crs_string='EPSG:3161',
            tile_size=(256, 256),
            tile_stride=(128, 128),
        )
        world_grid = grid_builder.build_grid('ref', grid_config)
        paths = artifacts.IngestionPaths(str(tmp_path))
        return world_grid, paths, str(img), str(lbl)

    return _create


# ----- collision policy tests
def test_collision_policy_skip(tmp_path, test_setup_factory):
    '''
    Given: An incumbent block pool with established catalog lineage.
    When: Ingesting an overlapping batch under policy 'skip'.
    Then: Preserve incumbent blocks and record skipped collisions.
    '''
    world_grid, paths, img, lbl = test_setup_factory()

    # run initial batch (run 1)
    rep1_fpath = str(tmp_path / 'rep1.json')
    logger1 = ingest.IngestionLogger(
        name='test_ingest',
        log_file=rep1_fpath,
        enable_file_log=False,
    )
    logger1.init_summary(run_id='run_0001')
    config1 = blocks.BlockBuildingParameters(
        image_fpath=img,
        label_fpath=lbl,
        dem_pad=8,
        ignore_index=255,
        harmonize_run_id='harm_0001',
        ingest_run_id='run_0001',
        collision_policy='skip',
    )
    blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config1,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger1,
        collisions_fpath=str(tmp_path / 'collisions_run1.json'),
    )

    # verify initial catalog lineage
    with open(paths.data_blocks.catalog, 'r', encoding='UTF-8') as f:
        catalog_run1 = json.load(f)
    assert len(catalog_run1) > 0
    first_key = next(iter(catalog_run1))
    assert catalog_run1[first_key]['harmonize_run_id'] == 'harm_0001'
    assert catalog_run1[first_key]['ingest_run_id'] == 'run_0001'
    orig_hash = catalog_run1[first_key]['sha_256']

    # run overlapping batch (run 2) with skip policy
    rep2_fpath = str(tmp_path / 'rep2.json')
    logger2 = ingest.IngestionLogger(
        name='test_ingest',
        log_file=rep2_fpath,
        enable_file_log=False,
    )
    logger2.init_summary(run_id='run_0002')
    config2 = blocks.BlockBuildingParameters(
        image_fpath=img,
        label_fpath=lbl,
        dem_pad=8,
        ignore_index=255,
        harmonize_run_id='harm_0002',
        ingest_run_id='run_0002',
        collision_policy='skip',
    )
    col2_fpath = str(tmp_path / 'collisions_run2.json')
    blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config2,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger2,
        collisions_fpath=col2_fpath,
    )

    # verify catalog was untouched and retains original lineage
    with open(paths.data_blocks.catalog, 'r', encoding='UTF-8') as f:
        catalog_run2 = json.load(f)
    assert catalog_run2[first_key]['harmonize_run_id'] == 'harm_0001'
    assert catalog_run2[first_key]['ingest_run_id'] == 'run_0001'
    assert catalog_run2[first_key]['sha_256'] == orig_hash

    # verify collisions manifest
    assert os.path.exists(col2_fpath)
    with open(col2_fpath, 'r', encoding='UTF-8') as f:
        manifest = json.load(f)
    assert manifest['collision_policy'] == 'skip'
    assert manifest['total_collided'] == len(catalog_run1)
    for record in manifest['collided_blocks']:
        assert record['action_taken'] == 'skipped'
        assert record['incumbent_ingest_run'] == 'run_0001'
        assert record['incumbent_harmonize_run'] == 'harm_0001'

    # verify telemetry report
    c_stats = logger2.summary['data_blocks']['collisions']
    assert c_stats['blocks_collided'] == len(catalog_run1)
    assert c_stats['blocks_skipped'] == len(catalog_run1)
    assert c_stats['blocks_overwritten'] == 0


def test_collision_policy_overwrite(tmp_path, test_setup_factory):
    '''
    Given: An incumbent block pool with established catalog lineage.
    When: Ingesting an overlapping batch under policy 'overwrite'.
    Then: Overwrite incumbent blocks and update lineage metadata.
    '''
    world_grid, paths, img, lbl = test_setup_factory()

    # run initial batch (run 1)
    logger1 = ingest.IngestionLogger(
        name='test_ingest',
        log_file=str(tmp_path / 'rep1.json'),
        enable_file_log=False,
    )
    logger1.init_summary(run_id='run_0001')
    config1 = blocks.BlockBuildingParameters(
        image_fpath=img,
        label_fpath=lbl,
        dem_pad=8,
        ignore_index=255,
        harmonize_run_id='harm_0001',
        ingest_run_id='run_0001',
        collision_policy='skip',
    )
    blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config1,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger1,
    )

    # run overlapping batch (run 2) with overwrite policy
    logger2 = ingest.IngestionLogger(
        name='test_ingest',
        log_file=str(tmp_path / 'rep2.json'),
        enable_file_log=False,
    )
    logger2.init_summary(run_id='run_0002')
    config2 = blocks.BlockBuildingParameters(
        image_fpath=img,
        label_fpath=lbl,
        dem_pad=8,
        ignore_index=255,
        harmonize_run_id='harm_0002',
        ingest_run_id='run_0002',
        collision_policy='overwrite',
    )
    col2_fpath = str(tmp_path / 'collisions_run2.json')
    blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config2,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger2,
        collisions_fpath=col2_fpath,
    )

    # verify catalog was updated with new lineage
    with open(paths.data_blocks.catalog, 'r', encoding='UTF-8') as f:
        catalog_run2 = json.load(f)
    first_key = next(iter(catalog_run2))
    assert catalog_run2[first_key]['harmonize_run_id'] == 'harm_0002'
    assert catalog_run2[first_key]['ingest_run_id'] == 'run_0002'

    # verify collisions manifest
    with open(col2_fpath, 'r', encoding='UTF-8') as f:
        manifest = json.load(f)
    assert manifest['collision_policy'] == 'overwrite'
    for record in manifest['collided_blocks']:
        assert record['action_taken'] == 'overwritten'
        assert record['incumbent_ingest_run'] == 'run_0001'
        assert record['incumbent_harmonize_run'] == 'harm_0001'

    # verify telemetry
    c_stats = logger2.summary['data_blocks']['collisions']
    assert c_stats['blocks_overwritten'] > 0
    assert c_stats['blocks_skipped'] == 0


def test_collision_policy_error(tmp_path, test_setup_factory):
    '''
    Given: An incumbent block pool with established catalog lineage.
    When: Ingesting an overlapping batch under policy 'error'.
    Then: Raise ArtifactError before creating or modifying blocks.
    '''
    world_grid, paths, img, lbl = test_setup_factory()

    # run initial batch (run 1)
    logger1 = ingest.IngestionLogger(
        name='test_ingest',
        log_file=str(tmp_path / 'rep1.json'),
        enable_file_log=False,
    )
    logger1.init_summary(run_id='run_0001')
    config1 = blocks.BlockBuildingParameters(
        image_fpath=img,
        label_fpath=lbl,
        dem_pad=8,
        ignore_index=255,
        harmonize_run_id='harm_0001',
        ingest_run_id='run_0001',
        collision_policy='skip',
    )
    blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config1,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger1,
    )

    # run overlapping batch (run 2) with error policy
    logger2 = ingest.IngestionLogger(
        name='test_ingest',
        log_file=str(tmp_path / 'rep2.json'),
        enable_file_log=False,
    )
    logger2.init_summary(run_id='run_0002')
    config2 = blocks.BlockBuildingParameters(
        image_fpath=img,
        label_fpath=lbl,
        dem_pad=8,
        ignore_index=255,
        harmonize_run_id='harm_0002',
        ingest_run_id='run_0002',
        collision_policy='error',
    )

    with pytest.raises(artifacts.ArtifactError, match='Intra-pool block'):
        blocks.run_blocks_building(
            world_grid,
            paths.data_blocks,
            config2,
            policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
            logger=logger2,
        )

    # verify catalog retains run 1 lineage
    with open(paths.data_blocks.catalog, 'r', encoding='UTF-8') as f:
        catalog = json.load(f)
    first_key = next(iter(catalog))
    assert catalog[first_key]['harmonize_run_id'] == 'harm_0001'
