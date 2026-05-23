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
Data preparation configurator
'''

# standard imports
import typing
# local imports
import landseg.configs as configs

class DataPreparationConfigurator:
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
        self._cfg.transform.output_dpath = (
            f'{experiment_root}/artifacts/{dataset_name}/transform'
        )
        # here we set pipeline to data-prepare
        self._cfg.pipeline.name = 'data-prepare'

    @property
    def running_root_config(self) -> configs.RootConfig:
        '''Validate and return the `RootConfig`,'''
        self._cfg.transform.validate()
        return self._cfg

    def set_partition(
        self,
        validation_blocks_ratio: float,
        test_holdout_blocks_ratio: float
    ) -> typing.Self:
        '''Set block ratios for validation and test holdout.'''
        self._cfg.transform.partition.val_ratio = validation_blocks_ratio
        self._cfg.transform.partition.test_ratio = test_holdout_blocks_ratio
        return self

    def set_scoring(
        self,
        reward_classes: dict[int, float]
    ) -> typing.Self:
        '''Set block scoring criteria.'''
        self._cfg.transform.scoring.reward = reward_classes
        return self

    def set_hydration(self) -> typing.Self:
        '''Set blocks hydration'''
        return self
