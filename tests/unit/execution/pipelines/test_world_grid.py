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

'''Unit tests for the world grid execution pipeline.'''

# standard imports
import json
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- `exec_world_grid` tests
def test_world_grid_pipeline_success(tmp_path, dummy_data_paths):
    '''
    Given: Valid extent reference raster and grid configuration.
    When: `exec_world_grid` is executed.
    Then: Produce canonical world grid JSON artifact on disk.
    '''
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    grid_cfg = cfg_schema.data.world_grid
    grid_cfg.mode = 'ref'
    grid_cfg.output_dpath = str(tmp_path / 'world_grids')
    grid_cfg.params.ref_fpath = dummy_data_paths.extent
    grid_cfg.params.crs_string = 'EPSG:3161'
    grid_cfg.params.tile_size = (256, 256)
    grid_cfg.params.tile_stride = (128, 128)

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    pipelines.exec_world_grid(config)

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

    # verify grid report artifact was generated
    report_fpath = os.path.join(
        str(tmp_path / 'world_grids'),
        'grid_report.json'
    )
    assert os.path.exists(report_fpath)
    with open(report_fpath, 'r', encoding='utf-8') as f:
        report_data = json.load(f)
    assert report_data['status'] == 'SUCCESS'
    assert report_data['grid']['grid_fpath'] == grid_fpath
    assert report_data['total_tiles'] == len(grid_data)
