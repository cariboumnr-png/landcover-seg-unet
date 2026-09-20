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
Data preparation configurator
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class DataPreparationConfigurator(configurators.BaseConfigurator):
    '''Configure data preparation.'''

    def __init__(
        self,
        experiment_root: str,
    ):
        super().__init__(experiment_root, 'data-prepare')

    def set_partition(
        self,
        validation_blocks_ratio: float,
        test_holdout_blocks_ratio: float
    ) -> typing.Self:
        '''Set block ratios for validation and test holdout.'''
        self._cfg.data.preparation.partition.val_ratio = validation_blocks_ratio
        self._cfg.data.preparation.partition.test_ratio = test_holdout_blocks_ratio
        return self

    def set_oversampling(
        self,
        target_head: str | None,
        reward_classes: dict[int, float]
    ) -> typing.Self:
        '''Set blocks hydration for reward classes in the target head'''
        self._cfg.data.preparation.catalog.focal_target = target_head
        self._cfg.data.preparation.scoring.reward = reward_classes
        return self

    def set_features(
        self,
        features: dict[str, str | list[str]] | None
    ) -> typing.Self:
        '''
        Set feature channel selection schemes or inline band lists.

        Example:
            configurator.set_features({
                'sentinel2': 'rgb_nir',
                'dem': 'all',
            })
        '''
        self._cfg.data.preparation.features = features or {}
        return self

    def set_targets(
        self,
        targets: dict[str, str | dict[str, typing.Any]] | None
    ) -> typing.Self:
        '''
        Set target reclassification schemes or inline mappings.

        Example:
            configurator.set_targets({
                'landcover': 'binary',
                'leadspc': 'raw',
            })
        '''
        self._cfg.data.preparation.targets = targets or {}
        return self

    def set_rebuild(self, rebuild: bool) -> typing.Self:
        '''Set whether to force rebuild preparation artifacts.'''
        self._cfg.data.preparation.rebuild = rebuild
        return self
