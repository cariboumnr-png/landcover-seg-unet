# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Data ingestion configurator
'''

# standard imports
import typing
# local imports
import landseg.configs as configs

class DataIngestionConfigurator:
    '''Configure data ingestion.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str
    ):
        '''Initialize the configurator'''

        self._cfg = configs.RootConfig() # with all default values
        # set output dirpaths
        self._cfg.execution.exp_root = experiment_root
        self._cfg.foundation.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/foundation'
        )
        # here we set pipeline to data-ingest
        self._cfg.pipeline.name = 'data-ingest'

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        self._cfg.foundation.validate()
        return self._cfg

    def set_grid(
        self,
        crs: str,
        reference_raster_fpath: str,
        tile_size: int,
        tile_overlap: int
    ) -> typing.Self:
        '''Set study extent and grid specs.'''
        self._cfg.foundation.grid.crs = crs
        self._cfg.foundation.grid.extent.filepath = reference_raster_fpath
        self._cfg.foundation.grid.tile_specs.size_row = tile_size
        self._cfg.foundation.grid.tile_specs.size_col = tile_size
        self._cfg.foundation.grid.tile_specs.overlap_row = tile_overlap
        self._cfg.foundation.grid.tile_specs.overlap_col = tile_overlap
        return self

    def set_domains(
        self,
        domains: list[tuple[str, int]]
    ) -> typing.Self:
        '''Set domain data source.'''
        for fpath, index_base in domains:
            self._cfg.foundation.domains.add_domain(fpath, index_base)
        return self

    def set_model_dev_data(
        self,
        model_dev_image: str,
        model_dev_label: str,
        data_config: str
    ) -> typing.Self:
        '''Set trainig data source.'''
        self._cfg.foundation.datablocks.filepaths.dev_image = model_dev_image
        self._cfg.foundation.datablocks.filepaths.dev_label = model_dev_label
        self._cfg.foundation.datablocks.filepaths.config = data_config
        return self

    def set_test_holdout_data(
        self,
        test_holdout_image: str,
        test_holdout_label: str
    ) -> typing.Self:
        '''Set test holdout data source.'''
        self._cfg.foundation.datablocks.filepaths.test_image = test_holdout_image
        self._cfg.foundation.datablocks.filepaths.test_label = test_holdout_label
        return self
