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

'''Unit tests for the data ingestion execution pipeline.'''

# standard imports
import os
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- pipeline execution
def test_data_ingest_pipeline_success(tmp_path, dummy_data_paths):
    # Compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # Override foundation fields
    grid_cfg = cfg_schema.foundation.grid
    grid_cfg.mode = 'ref'
    grid_cfg.crs = 'EPSG:2958'
    grid_cfg.extent.filepath = dummy_data_paths['ref_fpath']
    grid_cfg.tile_specs.size_row = 256
    grid_cfg.tile_specs.size_col = 256
    grid_cfg.tile_specs.overlap_row = 128
    grid_cfg.tile_specs.overlap_col = 128

    blocks_cfg = cfg_schema.foundation.datablocks
    blocks_cfg.name = 'test_ingest_run'
    blocks_cfg.filepaths.dev_image = dummy_data_paths['dev_image']
    blocks_cfg.filepaths.dev_label = dummy_data_paths['dev_label']
    blocks_cfg.filepaths.test_image = dummy_data_paths['test_image']
    blocks_cfg.filepaths.test_label = dummy_data_paths['test_label']
    blocks_cfg.filepaths.config = dummy_data_paths['dataset_config']

    cfg_schema.foundation.output_dpath = str(tmp_path / 'foundation')
    cfg_schema.foundation.rebuild = True

    # Convert back to standard typed RootConfig dataclass
    config = omegaconf.OmegaConf.to_object(cfg_schema)

    # Run the ingestion pipeline
    pipelines.ingest(config)

    # Verify the generated outputs
    out_dpath = config.foundation.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'model_dev', 'catalog.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'test_holdout', 'catalog.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'ingest_report.json')
    )
