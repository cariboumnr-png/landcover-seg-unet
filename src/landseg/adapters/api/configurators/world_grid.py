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
World grid canonical lifecycle configurator.
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators


class WorldGridConfigurator(configurators.BaseConfigurator):
    '''Configure canonical world grid generation and persistence.'''

    def __init__(self, experiment_root: str):
        super().__init__(experiment_root, 'world-grid')

    def set_grid(
        self,
        tile_size: int = 256,
        tile_stride: int = 0,
        crs: str = '',
        mode: str = 'ref',
        reference_raster: str | None = None,
    ) -> typing.Self:
        '''Set study extent and grid specs.'''
        self._cfg.data.world_grid.mode = mode
        self._cfg.data.world_grid.manual_specs.crs_string = crs
        self._cfg.data.world_grid.tile_size = (tile_size, tile_size)
        self._cfg.data.world_grid.tile_stride = (
            tile_stride,
            tile_stride,
        )
        if reference_raster:
            self._cfg.data.world_grid.ref_fpath = reference_raster
        return self

    def set_output_dpath(self, output_dpath: str) -> typing.Self:
        '''Set output directory path for world grid artifacts.'''
        self._cfg.data.world_grid.output_dpath = output_dpath
        return self
