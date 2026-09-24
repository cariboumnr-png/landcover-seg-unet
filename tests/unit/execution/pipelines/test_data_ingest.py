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

'''Unit tests for the data ingestion execution pipeline.'''

# standard imports
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- pipeline execution
def test_data_ingest_pipeline_success(tmp_path, dummy_data_paths):
    # compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # override world grid fields
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    cfg_schema.data.harmonization.dataset_manifest = dummy_data_paths.manifest
    cfg_schema.data.harmonization.output_dpath = str(tmp_path / 'harmonized')

    cfg_schema.data.ingestion.output_dpath = str(tmp_path / 'ingested_data')
    cfg_schema.data.ingestion.rebuild = True

    # convert back to standard typed `RootConfig` dataclass
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # run world-grid and harmonize first to generate matching CRS rasters
    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)

    # run the ingestion pipeline
    pipelines.exec_ingest_data(config)

    # verify the generated outputs
    out_dpath = config.data.ingestion.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'catalog.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'schema.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'blocks')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'run_0001', 'ingest_report.json')
    )


def test_data_ingest_pipeline_targeted_harmonization_run(
    tmp_path,
    dummy_data_paths
):
    '''
    Given: Multiple harmonization run directories.
    When: `data.ingestion.harmonization_run` is set to 1.
    Then: Ingestion targets run_0001 output artifacts.
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
    cfg_schema.data.ingestion.rebuild = True
    cfg_schema.data.ingestion.harmonization_run = 1

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # run world-grid first, then harmonization twice to create run_0001 and run_0002
    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)
    pipelines.exec_harmonize_data(config)

    h_root = str(tmp_path / 'harmonized')
    assert os.path.exists(os.path.join(h_root, 'run_0001'))
    assert os.path.exists(os.path.join(h_root, 'run_0002'))

    # ingest targeting run_0001
    pipelines.exec_ingest_data(config)
    out_dpath = config.data.ingestion.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'catalog.json')
    )


def test_data_ingest_pipeline_with_spectral_and_topo(
    tmp_path,
    dummy_data_paths
):
    '''
    Given: Pipeline config with add_topo and add_spectral configured.
    When: Executing data ingestion pipeline.
    Then: Materialize data blocks with derived feature channels.
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
    cfg_schema.data.ingestion.rebuild = True
    cfg_schema.data.ingestion.datablocks.add_topo = ['slope']
    cfg_schema.data.ingestion.datablocks.add_spectral = ['ndvi']

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)
    pipelines.exec_ingest_data(config)

    # verify generated outputs
    out_dpath = config.data.ingestion.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'catalog.json')
    )


def test_data_ingest_pipeline_multi_batch_catchup_and_idempotence(
    tmp_path,
    dummy_data_paths
):
    '''
    Given: Two completed harmonization runs and no prior ingestion.
    When: `exec_ingest_data` runs in auto pending mode, then runs again.
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

    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)
    pipelines.exec_harmonize_data(config)

    # first ingestion: should ingest both run_0001 and run_0002
    pipelines.exec_ingest_data(config)

    out_dpath = config.data.ingestion.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'run_0001', 'ingest_report.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'run_0002', 'ingest_report.json')
    )
    assert not os.path.exists(os.path.join(out_dpath, 'run_0003'))

    # second ingestion: should detect 0 pending runs and cleanly no-op
    pipelines.exec_ingest_data(config)
    assert not os.path.exists(os.path.join(out_dpath, 'run_0003'))
