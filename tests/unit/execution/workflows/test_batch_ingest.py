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

'''Unit tests for batch ingestion workflow (batch_ingest.py).'''

# standard imports
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows.batch_ingest as batch_ingest


# ----- `execute_batch_ingest` workflow tests
def test_batch_ingest_multi_batch_catchup_and_idempotence(
    tmp_path,
    dummy_data_paths,
):
    '''
    Given: Two completed harmonization runs and no prior ingestion.
    When: `execute_batch_ingest` runs in auto pending mode, then runs again.
    Then: Both batches are ingested sequentially, and re-run is a no-op.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    cfg_schema.data.harmonization.dataset_manifest = dummy_data_paths.manifest
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')
    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested_data')
    cfg_schema.data.ingestion.rebuild = False
    cfg_schema.data.ingestion.harmonization_run = None

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.WorldGridGeneration(config).run()
    pipelines.DataHarmonization(config).run()

    # run batch 2 with distinct resampling config to avoid run collision
    cfg_schema.data.harmonization.resampling_continuous = 'nearest'
    config_batch2 = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )
    pipelines.DataHarmonization(config_batch2).run()

    # first ingestion: should ingest both run_0001 and run_0002
    batch_ingest.execute_batch_ingest(config)

    out_dpath = config.data.ingestion.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'run_0001', 'report.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'run_0002', 'report.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'run_0002', 'collisions.json')
    )
    assert not os.path.exists(os.path.join(out_dpath, 'run_0003'))

    # second ingestion: should detect 0 pending runs and cleanly no-op
    batch_ingest.execute_batch_ingest(config)
    assert not os.path.exists(os.path.join(out_dpath, 'run_0003'))
