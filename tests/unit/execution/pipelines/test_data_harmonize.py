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

'''Unit tests for the data harmonization execution pipeline.'''

# standard imports
import json
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- `exec_harmonize_data` tests
def test_data_harmonize_pipeline_success(tmp_path, dummy_data_paths):
    '''
    Given: Valid dummy input rasters and manifest configuration.
    When: `exec_harmonize_data` is executed.
    Then: Produce harmonized VRTs, valid pixel mask, and canonical world grid.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # configure world grid and harmonization
    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    harm_cfg = cfg_schema.data.harmonization
    harm_cfg.dataset_manifest = dummy_data_paths.manifest
    harm_cfg.output_dpath = str(tmp_path / 'harmonized')

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)

    # verify run folder output
    run_dir = str(tmp_path / 'harmonized' / 'run_0001')
    assert os.path.exists(run_dir)
    assert os.path.exists(os.path.join(run_dir, 'harmonize_report.json'))
    assert os.path.exists(os.path.join(run_dir, 'valid_pixel_mask.vrt'))

    # verify canonical world grid artifact was generated
    grid_fpath = os.path.join(
        str(tmp_path / 'world_grids'),
        'grid_row_256_128_col_256_128.json'
    )
    assert os.path.exists(grid_fpath)
    with open(grid_fpath, 'r', encoding='utf-8') as f:
        grid_data = json.load(f)
    assert isinstance(grid_data, list)
    assert len(grid_data) > 0


def test_data_harmonize_pipeline_collision_skip(tmp_path, dummy_data_paths):
    '''
    Given: An already completed harmonization run.
    When: `exec_harmonize_data` is executed again with identical inputs.
    Then: Second run is detected as collision and recorded as SKIPPED.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    harm_cfg = cfg_schema.data.harmonization
    harm_cfg.dataset_manifest = dummy_data_paths.manifest
    harm_cfg.output_dpath = str(tmp_path / 'harmonized')

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.exec_world_grid(config)
    pipelines.exec_harmonize_data(config)
    pipelines.exec_harmonize_data(config)

    manifest_fp = os.path.join(
        str(tmp_path / 'harmonized'), 'harmonization_runs.json'
    )
    ctrl = artifacts.Controller[dict](manifest_fp)
    runs = ctrl.fetch()
    assert len(runs) == 2
    uids = list(runs.keys())
    assert runs[uids[0]]['status'] == 'SUCCESS'
    assert runs[uids[1]]['status'] == 'SKIPPED'

