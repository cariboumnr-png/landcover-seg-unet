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

'''Integration tests for continuous end-to-end data intake workflow.'''

# standard imports
import typing
# third-party imports
import omegaconf
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines
import landseg.execution.workflows.e2e_intake as e2e_intake


# ----- `execute_e2e_intake` workflow integration tests


def test_e2e_intake_workflow_success(tmp_path, dummy_data_paths):
    '''
    Given: Canonical raw raster inputs on the filesystem.
    When: Running `execute_e2e_intake` workflow.
    Then: Harmonize rasters and ingest into canonical block pool.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # world grid configuration
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.ref_fpath = dummy_data_paths.extent
    grid_cfg.tile_size = (256, 256)
    grid_cfg.tile_stride = (128, 128)

    # harmonization configuration
    harm_cfg = cfg_schema.data.harmonization
    harm_cfg.dataset_manifest = dummy_data_paths.manifest
    harm_cfg.output_dpath = str(tmp_path / 'harmonized')

    # ingestion configuration
    ingest_cfg = cfg_schema.data.ingestion
    ingest_cfg.output_dpath = str(tmp_path / 'ingested_data')
    ingest_cfg.rebuild = False
    ingest_cfg.harmonization_run = None

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # 1. establish world grid canvas
    pipelines.WorldGridGeneration(config).run()

    # 2. execute end-to-end data intake workflow
    e2e_intake.execute_e2e_intake(config)

    # verify harmonization artifacts
    h_run = tmp_path / 'harmonized' / 'run_0001'
    assert (h_run / 'report.json').exists()
    assert (h_run / 'valid_pixel_mask.vrt').exists()

    # verify ingestion artifacts
    i_root = tmp_path / 'ingested_data'
    assert (i_root / 'run_0001' / 'report.json').exists()
    assert (i_root / 'data_blocks' / 'catalog.json').exists()
    assert (i_root / 'data_blocks' / 'schema.json').exists()
    catalog_fpath =  i_root / 'data_blocks' / 'catalog.json'
    catalog = artifacts.Controller[dict](catalog_fpath).fetch()
    assert catalog
    assert len(catalog) > 0
