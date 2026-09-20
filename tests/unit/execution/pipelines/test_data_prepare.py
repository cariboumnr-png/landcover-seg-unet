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

'''Unit tests for the data preparation execution pipeline.'''

# standard imports
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- pipeline execution
def test_data_prepare_pipeline_success(tmp_path, dummy_data_paths):
    '''
    Given: A RootConfig pointing to valid raster inputs and temporary
        output directories.
    When: Running data ingest followed by data prepare pipelines.
    Then: Correctly partition the data blocks, aggregate image stats,
        normalize blocks, and compile schemas.
    '''
    # compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # override world grid fields
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    cfg_schema.data.harmonization.dataset_manifest = dummy_data_paths.manifest
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')

    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested')
    cfg_schema.data.ingestion.rebuild = True

    # override transform fields
    transform_cfg = cfg_schema.data.preparation
    transform_cfg.output_dpath = str(tmp_path / 'prepared')
    transform_cfg.rebuild = True

    transform_cfg.catalog.valid_pxs = {'image': 0.05}
    transform_cfg.catalog.focal_target = None

    transform_cfg.partition.val_ratio = 0.2
    transform_cfg.partition.test_ratio = 0.1
    transform_cfg.partition.buffer_step = 1

    transform_cfg.scoring.reward = {0: 1.0}
    transform_cfg.scoring.alpha = 1.0
    transform_cfg.scoring.beta = 0.5

    transform_cfg.hydration.max_skew_rate = 1.5

    # convert back to standard typed `RootConfig` dataclass
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # 1) run world-grid and harmonize to populate ETL outputs in EPSG:3161
    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)

    # 2) run the ingestion pipeline to build ingestion inputs
    pipelines.exec_ingest_data(config)

    # 3) run the preparation pipeline
    pipelines.exec_prepare_data(config)

    # verify the generated preparation outputs
    out_dpath = config.data.preparation.output_dpath
    assert os.path.exists(os.path.join(out_dpath, 'block_splits_source.json'))
    assert os.path.exists(
        os.path.join(out_dpath, 'block_splits_prepared.json')
    )
    assert os.path.exists(os.path.join(out_dpath, 'image_stats.json'))
    assert os.path.exists(os.path.join(out_dpath, 'prep_report.json'))
    assert os.path.exists(os.path.join(out_dpath, 'schema.json'))
