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
Data harmonization ETL configurator
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators


class DataHarmonizationConfigurator(configurators.BaseConfigurator):
    '''Configure data harmonization ETL.'''

    def __init__(
        self,
        experiment_root: str,
    ):
        super().__init__(experiment_root, 'data-harmonize')

    def set_dataset_manifest(
        self,
        dataset_manifest: str,
    ) -> typing.Self:
        '''Set dataset metadata manifest path.'''
        self._cfg.data.harmonization.dataset_manifest = dataset_manifest
        return self

    def set_resampling(
        self,
        continuous: str = 'bilinear',
        categorical: str = 'nearest',
    ) -> typing.Self:
        '''Set raster resampling methods.'''
        self._cfg.data.harmonization.resampling_continuous = continuous
        self._cfg.data.harmonization.resampling_categorical = categorical
        return self

    def set_grid(
        self,
        tile_size: int = 256,
        tile_stride: int = 0,
        output_dpath: str | None = None,
    ) -> typing.Self:
        '''Set target world grid tile dimensions.'''
        self._cfg.data.world_grid.params.tile_size = (tile_size, tile_size)
        self._cfg.data.world_grid.params.tile_stride = (
            tile_stride,
            tile_stride,
        )
        if output_dpath:
            self._cfg.data.world_grid.output_dpath = output_dpath
        return self

    def set_output_dpath(self, output_dpath: str) -> typing.Self:
        '''Set output directory path for harmonized artifacts.'''
        self._cfg.data.harmonization.output_dpath = output_dpath
        return self

    def set_grid_reference(self, output_dpath: str) -> typing.Self:
        '''
        Set directory path containing upstream world grid artifacts.

        Args:
            output_dpath:
                Directory containing the canonical world grid report.
        '''
        self._cfg.data.world_grid.output_dpath = output_dpath
        return self
