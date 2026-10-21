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

'''
Logging utilities for world grid generation pipeline execution.

This module provides a specialized `Logger` subclass that tracks world
grid construction and persists a structured JSON run report at shutdown.

Public APIs:
    - `GridLogger`: Logger tracking world grid execution and report JSON.
'''

# standard imports
import datetime
import pathlib
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.geopipe.contracts.grid as contracts
import landseg.geopipe.core as geo_core
import landseg.utils as utils


# ----- public classes
class GridLogger(utils.Logger):
    '''
    A specialized `Logger` wrapper that tracks world grid generation
    and persists a structured JSON report at shutdown.
    '''

    def __init__(self, *args: typing.Any, **kwargs: typing.Any):
        '''Initialize the GridLogger instance.'''
        super().__init__(*args, **kwargs)
        self.summary: contracts.GridReportSchema | None = None

    def init_summary(
        self,
        *,
        run_id: str = '',
        timestamp: str | None = None
    ) -> None:
        '''Initialize the structured run report summary.'''
        t = timestamp or datetime.datetime.now().strftime(c.TF_ISO8601)
        self.summary = {
            'run_id': run_id,
            'timestamp': t,
            'status': 'SUCCESS',
            'grid': {
                'grid_fpath': '',
                'grid_id': '',
                'crs': '',
                'pixel_size': (0.0, 0.0),
                'tile_size': (0, 0),
                'tile_overlap': (0, 0),
            },
            'total_tiles': 0,
        }

    def set_grid_report(
        self,
        world_grid: geo_core.GridLayout,
        grid_filepath: pathlib.Path,
    ) -> None:
        '''Record world grid summary report and tile count.'''
        grid_report: contracts.WorldGridReport = {
            'grid_fpath': self.abs_path(grid_filepath),
            'grid_id': world_grid.gid,
            'crs': world_grid.crs,
            'pixel_size': world_grid.pixel_size,
            'tile_size': world_grid.tile_size,
            'tile_overlap': world_grid.tile_overlap,
        }
        if self.summary is not None:
            self.summary['grid'] = grid_report
            self.summary['total_tiles'] = len(world_grid)

    def set_status(self, status: typing.Literal['SUCCESS', 'FAILED']) -> None:
        '''Update the execution status of the pipeline run.'''
        if self.summary is not None:
            self.summary['status'] = status

    def close(self) -> None:
        '''Flush and close file handlers, writing report JSON if set.'''
        if self.summary is not None and self.log_file:
            artifacts.Controller[contracts.GridReportSchema](
                self.log_file
            ).persist(self.summary)
        super().close()
