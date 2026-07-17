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
    When: Running the data ingest followed by the data prepare pipelines.
    Then: Correctly partition the data blocks, aggregate image stats,
        normalize blocks, and compile schemas.
    '''
    # compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # override foundation grid fields
    grid_cfg = cfg_schema.foundation.grid
    grid_cfg.mode = 'ref'
    grid_cfg.crs = 'EPSG:2958'
    grid_cfg.extent.filepath = dummy_data_paths.extent
    grid_cfg.tile_specs.size_row = 256
    grid_cfg.tile_specs.size_col = 256
    grid_cfg.tile_specs.overlap_row = 128
    grid_cfg.tile_specs.overlap_col = 128

    # override foundation datablocks fields
    blocks_cfg = cfg_schema.foundation.datablocks
    blocks_cfg.name = 'test_prepare_run'
    blocks_cfg.filepaths.dev_image = dummy_data_paths.dev_image
    blocks_cfg.filepaths.dev_label = dummy_data_paths.dev_label
    blocks_cfg.filepaths.test_image = dummy_data_paths.test_image
    blocks_cfg.filepaths.test_label = dummy_data_paths.test_label
    blocks_cfg.filepaths.config = dummy_data_paths.config

    cfg_schema.foundation.output_dpath = str(tmp_path / 'foundation')
    cfg_schema.foundation.rebuild = True

    # override transform fields
    transform_cfg = cfg_schema.transform
    transform_cfg.output_dpath = str(tmp_path / 'transform')
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

    # 1) run the ingestion pipeline to build foundation inputs
    pipelines.ingest(config)

    # 2) run the preparation pipeline
    pipelines.prepare(config)

    # verify the generated transform outputs
    out_dpath = config.transform.output_dpath
    assert os.path.exists(os.path.join(out_dpath, 'block_splits_source.json'))
    assert os.path.exists(os.path.join(out_dpath, 'block_splits_transformed.json'))
    assert os.path.exists(os.path.join(out_dpath, 'image_stats.json'))
    assert os.path.exists(os.path.join(out_dpath, 'prep_report.json'))
    assert os.path.exists(os.path.join(out_dpath, 'schema.json'))
