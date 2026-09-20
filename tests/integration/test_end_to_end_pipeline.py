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

# pylint: disable=missing-function-docstring,too-many-statements,duplicate-code

'''Integration tests for the end-to-end data pipeline lifecycle.'''

# standard imports
import json
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- end-to-end pipeline execution
def test_end_to_end_data_pipeline_lifecycle(tmp_path, dummy_data_paths):
    '''
    Given: Raw raster datasets and manifest definitions.
    When: Executing harmonize, ingest, and prepare sequentially.
    Then: Produce canonical harmonized rasters and world grid,
        unified ingested datablocks, and experiment partition splits.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # 1. World grid configuration
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    # 2. Harmonization configuration
    harm_cfg = cfg_schema.data.harmonization
    harm_cfg.dataset_manifest = dummy_data_paths.manifest
    harm_cfg.output_dpath = str(tmp_path / 'harmonized')

    # 3. Ingestion configuration
    ingest_cfg = cfg_schema.data.ingestion
    ingest_cfg.output_dpath = str(tmp_path / 'ingested_data')
    ingest_cfg.rebuild = True

    # 4. Preparation configuration
    prep_cfg = cfg_schema.data.preparation
    prep_cfg.output_dpath = str(tmp_path / 'prepared_data')
    prep_cfg.rebuild = True
    prep_cfg.catalog.valid_pxs = {'image': 0.05}
    prep_cfg.catalog.focal_target = None
    prep_cfg.partition.val_ratio = 0.2
    prep_cfg.partition.test_ratio = 0.1
    prep_cfg.partition.buffer_step = 1
    prep_cfg.scoring.reward = {0: 1.0}
    prep_cfg.scoring.alpha = 1.0
    prep_cfg.scoring.beta = 0.5
    prep_cfg.hydration.max_skew_rate = 1.5

    # convert to typed RootConfig
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # ----- stage 1: world grid
    pipelines.exec_world_grid(config)
    assert os.path.exists(
        os.path.join(
            str(tmp_path / 'world_grids'),
            'grid_row_256_128_col_256_128.json'
        )
    )

    # ----- stage 2: harmonize
    pipelines.exec_harmonize_data(config)

    h_run = os.path.join(config.data.harmonization.output_dpath, 'run_0001')
    assert os.path.exists(os.path.join(h_run, 'harmonize_report.json'))
    assert os.path.exists(
        os.path.join(h_run, 'harmonized_features_STACKED.vrt')
    )
    assert os.path.exists(
        os.path.join(h_run, 'harmonized_labels_STACKED.vrt')
    )
    assert os.path.exists(os.path.join(h_run, 'valid_pixel_mask.vrt'))

    # ----- stage 3: ingest
    pipelines.exec_ingest_data(config)

    i_root = config.data.ingestion.output_dpath
    assert os.path.exists(os.path.join(i_root, 'ingest_report.json'))
    assert os.path.exists(
        os.path.join(i_root, 'data_blocks', 'catalog.json')
    )
    assert os.path.exists(
        os.path.join(i_root, 'data_blocks', 'schema.json')
    )
    assert os.path.exists(
        os.path.join(i_root, 'data_blocks', 'blocks')
    )

    # verify catalog contents
    with open(
        os.path.join(i_root, 'data_blocks', 'catalog.json'),
        'r',
        encoding='utf-8'
    ) as f:
        catalog = json.load(f)
    assert len(catalog) > 0

    # ----- stage 4: prepare
    pipelines.exec_prepare_data(config)

    p_root = config.data.preparation.output_dpath
    assert os.path.exists(os.path.join(p_root, 'prep_report.json'))
    assert os.path.exists(
        os.path.join(p_root, 'block_splits_source.json')
    )
    assert os.path.exists(
        os.path.join(p_root, 'block_splits_prepared.json')
    )
    assert os.path.exists(os.path.join(p_root, 'image_stats.json'))
    assert os.path.exists(os.path.join(p_root, 'schema.json'))

    # verify splits contain train and val
    with open(
        os.path.join(p_root, 'block_splits_source.json'),
        'r',
        encoding='utf-8'
    ) as f:
        splits = json.load(f)
    assert 'train' in splits
    assert 'val' in splits
    assert len(splits['train']) > 0
