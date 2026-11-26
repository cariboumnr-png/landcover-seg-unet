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

'''Data preparation configurator.'''

# standard imports
import pathlib
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class DataPreparationConfigurator(configurators.BaseConfigurator):
    '''Configure data preparation.'''

    def __init__(
        self,
        *,
        upstream_dpath: str | pathlib.Path | None = None,
        output_dpath: str | pathlib.Path | None = None,
        experiment_root: str | pathlib.Path | None = None,
    ):
        '''
        Initialize the data preparation configurator.

        Args:
            upstream_dpath:
                Optional directory path for upstream ingestion outputs.
            output_dpath:
                Optional destination directory for prepared artifacts.
            experiment_root:
                Optional experiment root directory for unified mode.
        '''
        super().__init__(
            'data-prepare',
            upstream_dpath=upstream_dpath,
            output_dpath=output_dpath,
            experiment_root=experiment_root,
        )

    def set_partition(
        self,
        validation_blocks_ratio: float,
        test_holdout_blocks_ratio: float
    ) -> typing.Self:
        '''
        Set block ratios for validation and test holdout splits.

        Args:
            validation_blocks_ratio:
                Fraction of blocks assigned to the validation split.
            test_holdout_blocks_ratio:
                Fraction of blocks assigned to the test holdout split.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        self.cfg.data.preparation.partition.val_ratio = validation_blocks_ratio
        self.cfg.data.preparation.partition.test_ratio = test_holdout_blocks_ratio
        return self

    def set_oversampling(
        self,
        target_head: str | None,
        reward_classes: dict[int, float]
    ) -> typing.Self:
        '''
        Set block hydration weights for reward classes in target head.

        Args:
            target_head:
                Optional focal target head name used for scoring.
            reward_classes:
                Mapping from class ID to oversampling reward weight.

        Returns:
            typing.Self:
                Configurator instance for method chaining.
        '''
        self.cfg.data.preparation.datasetview.focal_target = target_head
        self.cfg.data.preparation.scoring.reward = reward_classes
        return self

    def set_features(
        self,
        features: dict[str, str | list[str]] | None
    ) -> typing.Self:
        '''
        Set feature channel selection schemes or inline band lists.

        Args:
            features:
                Optional mapping from raster source name to preset
                scheme name or explicit list of band identifiers.

        Returns:
            typing.Self:
                Configurator instance for method chaining.

        Example:
            configurator.set_features({
                'sentinel2': 'rgb_nir',
                'dem': 'all',
            })
        '''
        self.cfg.data.preparation.datasetview.features = features or {}
        return self

    def set_targets(self, targets: dict[str, typing.Any] | None) -> typing.Self:
        '''
        Set target reclassification schemes or inline mappings.

        Args:
            targets:
                Optional mapping from target head name to preset scheme
                name or inline reclassification mapping.

        Returns:
            typing.Self:
                Configurator instance for method chaining.

        Example:
            configurator.set_targets({
                'landcover': 'binary',
                'leadspc': 'raw',
            })
        '''
        self.cfg.data.preparation.datasetview.targets = targets or {}
        return self

    def set_rebuild(self, rebuild: bool) -> typing.Self:
        '''Set whether to force rebuild preparation artifacts.'''
        self.cfg.data.preparation.rebuild = rebuild
        return self
