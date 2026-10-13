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

'''Configuration for `geopipe.preparation` module.'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


@dataclasses.dataclass
class DatasetViewConfig(base.BaseConfigSection):
    '''Configuration for `geopipe.preparation.dataset` module.'''
    valid_pxs: dict[str, float] = field(default_factory=lambda: {'image': 0.9})
    focal_target: str | None = None
    test_catalog: str | None = None
    non_overlapping_test_grid: bool = True
    features: dict[str, typing.Any] = field(default_factory=dict)
    targets: dict[str, typing.Any] = field(default_factory=dict)

    def validate(self) -> None:
        for k, v in self.valid_pxs.items():
            self.require_type_range(k, 'array_key', str)
            self.require_type_range(v, f'{k} valid threshold', float, (0, 1))

        for k, v in self.features.items():
            self.require_type_range(k, 'feature_config_name', (str))
            self.require_type_range(v, 'feature_config', (str, list))

        for k, v in self.targets.items():
            self.require_type_range(k, 'target_config_name', str)
            self.require_type_range(v, 'target_config', dict)


@dataclasses.dataclass
class Partition(base.BaseConfigSection):
    '''Configuration for `geopipe.preparation.partition` module.'''
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    buffer_step: int = 1
    train_aoi: str | None = None
    val_aoi: str | None = None
    test_aoi: str | None = None
    aoi_min_overlap: float = 0.5

    def validate(self) -> None:
        self.require_attr_type_range('val_ratio', float, (0, 1))
        self.require_attr_type_range('test_ratio', float, (0, 1))
        self.require_attr_type_range('aoi_min_overlap', float, (0, 1))


@dataclasses.dataclass
class Scoring(base.BaseConfigSection):
    '''Configuration for `geopipe.preparation.partition` module.'''
    reward: dict[int, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 0.0

    def validate(self) -> None:
        self.require_attr_type_range('alpha', float, (0, None))
        self.require_attr_type_range('beta', float, (0, None))


@dataclasses.dataclass
class Hydration(base.BaseConfigSection):
    '''Configuration for `geopipe.preparation.partition` module.'''
    max_skew_rate: float = 10.0

    def validate(self) -> None:
        self.require_attr_type_range('max_skew_rate', float, (0, None))


@dataclasses.dataclass
class DataPreparationConfig(base.BaseConfigSection):
    '''Configuration for `geopipe.preparation` module.'''
    datasetview: DatasetViewConfig = field(default_factory=DatasetViewConfig)
    partition: Partition = field(default_factory=Partition)
    scoring: Scoring = field(default_factory=Scoring)
    hydration: Hydration = field(default_factory=Hydration)
    rebuild: bool = False
    output_dpath: str = '${execution.exp_root}/artifacts/prepared_data'

    def validate(self) -> None:
        self.datasetview.validate()
        self.partition.validate()
        self.scoring.validate()
        self.hydration.validate()
