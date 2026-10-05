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

'''
World grid pipeline command implementation.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.execution.pipelines.base as base
import landseg.geopipe.contracts as contracts
import landseg.geopipe.grid as grid


# ----- public classes
class WorldGridGeneration(base.GeoPipeline):
    '''World grid generation pipeline.'''

    pipeline_name: str = 'world-grid'
    context: None
    logger: grid.GridLogger
    pipeline_paths: artifacts.WorldGridPaths

    @property
    def grid_cfg(self):
        '''Return world grid configuration.'''
        return self.config.data.world_grid

    def run(self) -> None:
        '''Execute the world-grid pipeline.'''
        self.validate()
        self._initialize_run()

        try:
            self.logger.log_sep()
            self.logger.log('INFO', 'Building/loading canonical world grid')

            is_loaded, grid_fp, world_grid = grid.prepare_world_grid(self.grid_cfg)
            status_str = 'loaded' if is_loaded else 'created and persisted'

            grid_report: contracts.WorldGridReport = {
                'grid_fpath': grid_fp,
                'grid_id': world_grid.gid,
                'crs': world_grid.crs,
                'pixel_size': world_grid.pixel_size,
                'tile_size': world_grid.tile_size,
                'tile_overlap': world_grid.tile_overlap,
            }
            self.logger.set_grid_report(grid_report, total_tiles=len(world_grid))

            self.logger.log('INFO', f'[COMPLETE] World grid {status_str}')
            self.logger.log('INFO', f'Grid ID: {world_grid.gid}')
            self.logger.log('INFO', f'Grid artifact file path: {grid_fp}')
            self.logger.log('INFO', f'CRS: {world_grid.crs}')
            self.logger.log('INFO', f'Total Tiles: {len(world_grid)}')
        except Exception as err:
            self.logger.set_summary_status('FAILED')
            self.logger.log('ERROR', f'World grid execution failed: {err}')
            raise
        finally:
            self.logger.log_sep()
            self.logger.close()

    def validate(self) -> None:
        '''Validate world-grid configuration and input references.'''
        if self.grid_cfg.mode == 'ref':
            ref_fp = self.grid_cfg.params.ref_fpath
            if not ref_fp or not os.path.exists(ref_fp):
                raise FileNotFoundError(
                    f'Reference raster for world-grid does not exist: {ref_fp}'
                )

    def _create_logger(self) -> grid.GridLogger:
        '''Instantiate and configure the grid logger.'''
        logger = grid.GridLogger(
            name=self.pipeline_name,
            log_file=self.pipeline_paths.report,
            enable_file_log=False,
        )
        logger.init_summary(run_id='world-grid')
        return logger
