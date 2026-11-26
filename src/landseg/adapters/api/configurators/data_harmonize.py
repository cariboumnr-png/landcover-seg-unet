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

'''Data harmonization ETL configurator.'''

# standard imports
import pathlib
import typing
# local imports
import landseg.adapters.api.configurators as configurators


class DataHarmonizationConfigurator(configurators.BaseConfigurator):
    '''Configure data harmonization ETL.'''

    def __init__(
        self,
        *,
        upstream_dpath: str | pathlib.Path | None = None,
        output_dpath: str | pathlib.Path | None = None,
        experiment_root: str | pathlib.Path | None = None,
    ):
        '''
        Initialize the data harmonization configurator.

        Args:
            upstream_dpath:
                Optional directory path for upstream world grid outputs.
            output_dpath:
                Optional destination directory for harmonized artifacts.
            experiment_root:
                Optional experiment root directory for unified mode.
        '''
        super().__init__(
            'data-harmonize',
            upstream_dpath=upstream_dpath,
            output_dpath=output_dpath,
            experiment_root=experiment_root,
        )

    def set_dataset_manifest(
        self,
        dataset_manifest: str | pathlib.Path,
    ) -> typing.Self:
        '''Set dataset metadata manifest path.'''
        self.cfg.data.harmonization.dataset_manifest = str(dataset_manifest)
        return self

    def set_resampling(
        self,
        continuous: str = 'bilinear',
        categorical: str = 'nearest',
    ) -> typing.Self:
        '''
        Set raster resampling methods.

        Args:
            continuous:
                Resampling method for continuous raster bands.
            categorical:
                Resampling method for categorical label rasters.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        self.cfg.data.harmonization.resampling_continuous = continuous
        self.cfg.data.harmonization.resampling_categorical = categorical
        return self

    def set_grid(
        self,
        tile_size: int = 256,
        tile_stride: int = 0,
        output_dpath: str | pathlib.Path | None = None,
    ) -> typing.Self:
        '''
        Set target world grid tile dimensions and output path.

        Args:
            tile_size:
                Square tile side length in pixels.
            tile_stride:
                Stride between adjacent tiles in pixels.
            output_dpath:
                Optional destination directory for world grid outputs.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        self.cfg.data.world_grid.tile_size = (tile_size, tile_size)
        self.cfg.data.world_grid.tile_stride = (tile_stride, tile_stride)
        if output_dpath:
            self.cfg.data.world_grid.output_dpath = str(output_dpath)
        return self
