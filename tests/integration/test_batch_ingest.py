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

'''Integration tests for batch ingestion workflow (batch_ingest.py).'''

# standard imports
import pathlib
import typing
# third-party imports
import omegaconf
# local imports
import landseg.artifacts as artifacts
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
    grid_cfg.ref_fpath = dummy_data_paths.extent
    grid_cfg.tile_size = (256, 256)
    grid_cfg.tile_stride = (128, 128)

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

    out_dpath = pathlib.Path(config.data.ingestion.output_dpath)
    assert (out_dpath / 'run_0001' / 'report.json').exists()
    assert (out_dpath / 'run_0002' / 'report.json').exists()
    assert (out_dpath / 'run_0002' / 'collisions.json').exists()
    assert not (out_dpath / 'run_0003').exists()

    # second ingestion: should detect 0 pending runs and cleanly no-op
    batch_ingest.execute_batch_ingest(config)
    assert not (out_dpath / 'run_0003').exists()


def test_data_ingest_targeted_harmonization_run(
    tmp_path,
    dummy_data_paths,
):
    '''
    Given: Multiple harmonization run directories.
    When: `data.ingestion.harmonization_run` is set to 1.
    Then: Ingestion targets run_0001 output artifacts.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.ref_fpath = dummy_data_paths.extent
    grid_cfg.tile_size = (256, 256)
    grid_cfg.tile_stride = (128, 128)

    cfg_schema.data.harmonization.dataset_manifest = dummy_data_paths.manifest
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')
    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested_data')
    cfg_schema.data.ingestion.rebuild = True
    cfg_schema.data.ingestion.harmonization_run = 1

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.WorldGridGeneration(config).run()
    pipelines.DataHarmonization(config).run()

    # run batch 2 with distinct resampling config to produce run_0002
    cfg_schema.data.harmonization.resampling_continuous = 'nearest'
    config_batch2 = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )
    pipelines.DataHarmonization(config_batch2).run()

    h_root = tmp_path / 'harmonized'
    assert (h_root / 'run_0001').exists()
    assert (h_root / 'run_0002').exists()

    # ingest targeting run_0001
    pipelines.DataIngestion(config).run()
    out_dpath = pathlib.Path(config.data.ingestion.output_dpath)
    assert (out_dpath / 'data_blocks' / 'catalog.json').exists()
    assert (out_dpath / 'run_0001' / 'report.json').exists()
    assert not (out_dpath / 'run_0002').exists()


def test_data_ingest_with_spectral_and_topo_derivation(
    tmp_path,
    dummy_data_paths,
):
    '''
    Given: Pipeline config with add_topo and add_spectral configured.
    When: Executing data ingestion pipeline.
    Then: Materialize data blocks with derived feature channels.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.ref_fpath = dummy_data_paths.extent
    grid_cfg.tile_size = (256, 256)
    grid_cfg.tile_stride = (128, 128)

    cfg_schema.data.harmonization.dataset_manifest = dummy_data_paths.manifest
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')
    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested_data')
    cfg_schema.data.ingestion.rebuild = True
    cfg_schema.data.ingestion.datablocks.add_topo = ['slope']
    cfg_schema.data.ingestion.datablocks.add_spectral = ['ndvi']

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.WorldGridGeneration(config).run()
    pipelines.DataHarmonization(config).run()
    pipelines.DataIngestion(config).run()

    out_dpath = pathlib.Path(config.data.ingestion.output_dpath)
    assert (out_dpath / 'data_blocks' / 'catalog.json').exists()
    assert (out_dpath / 'data_blocks' / 'schema.json').exists()
    schema = artifacts.Controller[dict](
        out_dpath / 'data_blocks' / 'schema.json'
    ).fetch()
    assert schema


def test_data_harmonize_collision_skip(tmp_path, dummy_data_paths):
    '''
    Given: An already completed harmonization run.
    When: `DataHarmonization.run` is executed again with identical inputs.
    Then: Second run is detected as collision and recorded as SKIPPED.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.ref_fpath = dummy_data_paths.extent
    grid_cfg.tile_size = (256, 256)
    grid_cfg.tile_stride = (128, 128)

    harm_cfg = cfg_schema.data.harmonization
    harm_cfg.dataset_manifest = dummy_data_paths.manifest
    harm_cfg.output_dpath = str(tmp_path / 'harmonized')

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.WorldGridGeneration(config).run()
    pipelines.DataHarmonization(config).run()
    pipelines.DataHarmonization(config).run()

    manifest_fp = (
        pathlib.Path(tmp_path / 'harmonized') / 'harmonization_runs.json'
    )
    ctrl = artifacts.Controller[dict](manifest_fp)
    runs = ctrl.fetch()
    assert runs
    assert len(runs) == 2
    uids = list(runs.keys())
    assert runs[uids[0]]['status'] == 'SUCCESS'
    assert runs[uids[1]]['status'] == 'SKIPPED'
