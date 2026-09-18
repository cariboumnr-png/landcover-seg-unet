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

'''Unit tests for harmonization context resolution.'''

# standard imports
import dataclasses
import os
# third-party imports
import pytest
# local imports
import landseg.geopipe.contracts as contracts
import landseg.geopipe.grid as grid
import landseg.geopipe.harmonize as harmonize


# ----- private types
@dataclasses.dataclass
class _Params:
    tile_size: tuple[int, int] = (8, 8)
    tile_stride: tuple[int, int] = (4, 4)
    ref_fpath: str | None = None
    crs_string: str | None = 'EPSG:32617'
    origin: tuple[float, float] | None = (0.0, 0.0)
    pixel_size: tuple[float, float] | None = (10.0, 10.0)
    extent_in_crs_units: tuple[float, float] | None = (160.0, 160.0)


@dataclasses.dataclass
class _WorldGridConfig:
    mode: str = 'manual'
    params: _Params = dataclasses.field(default_factory=_Params)
    output_dpath: str = ''


# ----- `HarmonizationContext` tests
def test_harmonization_context_frozen(tmp_path):
    '''
    Given: Instantiated HarmonizationContext.
    When: Attempting to mutate an attribute.
    Then: Raise FrozenInstanceError.
    '''
    config = _WorldGridConfig(output_dpath=str(tmp_path))
    _, grid_fp, world_grid = grid.prepare_world_grid(config)
    ctx = harmonize.HarmonizationContext(
        grid=world_grid,
        grid_id=world_grid.gid,
        grid_fpath=grid_fp,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        # pylint: disable=dataclass-cannot-be-modified
        ctx.grid_id = 'new_id'


def test_build_harmonization_context_from_report_fpath(tmp_path):
    '''
    Given: Persisted world grid and grid report JSON artifact.
    When: `build_harmonization_context` is called with file path.
    Then: Successfully load context container with matching grid.
    '''
    config = _WorldGridConfig(output_dpath=str(tmp_path))
    _, grid_fp, world_grid = grid.prepare_world_grid(config)

    report_fp = os.path.join(str(tmp_path), 'grid_report.json')
    logger = grid.GridLogger(
        name='world-grid',
        log_file=report_fp,
        enable_file_log=False,
    )
    logger.init_summary(run_id='world-grid')
    grid_report: contracts.WorldGridReport = {
        'grid_fpath': grid_fp,
        'grid_id': world_grid.gid,
        'crs': world_grid.crs,
        'pixel_size': world_grid.pixel_size,
        'tile_size': world_grid.tile_size,
        'tile_overlap': world_grid.tile_overlap,
    }
    logger.set_grid_report(grid_report, total_tiles=len(world_grid))
    logger.close()

    ctx = harmonize.build_harmonization_context(report_fp)
    assert ctx.grid_id == world_grid.gid
    assert ctx.grid_fpath == grid_fp
    assert ctx.grid.gid == world_grid.gid


def test_build_harmonization_context_from_directory(tmp_path):
    '''
    Given: Persisted world grid and grid report JSON in directory.
    When: `build_harmonization_context` is called with directory path.
    Then: Resolve report file and return loaded context container.
    '''
    config = _WorldGridConfig(output_dpath=str(tmp_path))
    _, grid_fp, world_grid = grid.prepare_world_grid(config)

    report_fp = os.path.join(str(tmp_path), 'grid_report.json')
    logger = grid.GridLogger(
        name='world-grid',
        log_file=report_fp,
        enable_file_log=False,
    )
    logger.init_summary(run_id='world-grid')
    grid_report: contracts.WorldGridReport = {
        'grid_fpath': grid_fp,
        'grid_id': world_grid.gid,
        'crs': world_grid.crs,
        'pixel_size': world_grid.pixel_size,
        'tile_size': world_grid.tile_size,
        'tile_overlap': world_grid.tile_overlap,
    }
    logger.set_grid_report(grid_report, total_tiles=len(world_grid))
    logger.close()

    ctx = harmonize.build_harmonization_context(str(tmp_path))
    assert ctx.grid_id == world_grid.gid
    assert ctx.grid_fpath == grid_fp
    assert ctx.grid.gid == world_grid.gid
