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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Data transform schema
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.utils as utils

# alias
field = dataclasses.field

# -------------------------------DATA  TRANSFORM-------------------------------
@dataclasses.dataclass
class _CatalogView:
    valid_pxs: dict[str, float] = field(default_factory=lambda: {'image': 0.9})
    focal_target: str | None = None
    non_overlapping_test_grid: bool = True

    def validate(self):
        for k, v in self.valid_pxs.items():
            utils.must_within(v, f'{k} valid threshold', 0, 1)

@dataclasses.dataclass
class _Partition:
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    buffer_step: int = 1

    def validate(self):
        utils.must_within(self.val_ratio, 'validation block ratio', 0, 1)
        utils.must_within(self.test_ratio, 'test holdout block ratio', 0, 1)

@dataclasses.dataclass
class _Scoring:
    reward: dict[int, float] = field(default_factory=dict)
    alpha: float = 1.0
    beta: float = 0.0

    def validate(self):
        utils.must_within(self.alpha, 'scoring alpha', 0)
        utils.must_within(self.beta, 'scoring beta', 0)

@dataclasses.dataclass
class _Hydration:
    max_skew_rate: float = 10.0

    def validate(self):
        utils.must_within(self.max_skew_rate, 'hydration skew ratio', 0)

# ----- composite
@dataclasses.dataclass
class DataTransform:
    # ----- internal
    catalog: _CatalogView = field(default_factory=_CatalogView)
    partition: _Partition = field(default_factory=_Partition)
    scoring: _Scoring = field(default_factory=_Scoring)
    hydration: _Hydration = field(default_factory=_Hydration)
    output_dpath: str = (
        '${execution.exp_root}/artifacts/'
        '${foundation.datablocks.name}/transform'
    )

    def validate(self):
        self.catalog.validate()
        self.partition.validate()
        self.scoring.validate()
        self.hydration.validate()
