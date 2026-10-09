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

# pylint: disable=missing-class-docstring

'''
Objective, loss, and regularization search spaces for study sweeps.
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


@dataclasses.dataclass
class LossBalanceSearchSpace(base.BaseConfigSection):
    focal_weight: tuple[float, float] = (0.0, 1.0)
    dice_weight: tuple[float, float] = (0.0, 1.0)


@dataclasses.dataclass
class LossAuxiliarySearchSpace(base.BaseConfigSection):
    spectral_weight: tuple[float, float] = (0.0, 1e-2)
    tv_weight: tuple[float, float] = (0.0, 1e-3)


@dataclasses.dataclass
class RegularizationSearchSpace(base.BaseConfigSection):
    consistency_lambda: tuple[float, float] = (0.0, 1.0)


@dataclasses.dataclass
class HeadWeightsSearchSpace(base.BaseConfigSection):
    logit_adjust_alpha: tuple[float, float] = (0.0, 2.0)


@dataclasses.dataclass
class MtlJointSearchSpace(base.BaseConfigSection):
    consistency_lambda: tuple[float, float] = (0.0, 1.0)
    logit_adjust_alpha: tuple[float, float] = (0.0, 2.0)


@dataclasses.dataclass
class HierarchySearchSpace(base.BaseConfigSection):
    consistency_lambda: tuple[float, float] = (0.0, 1.0)
    consistency_reduction: list[str] = field(
        default_factory=lambda: ['mean', 'sum']
    )


@dataclasses.dataclass
class ObjectivesSearchConfig(base.BaseConfigSection):
    loss_balance: LossBalanceSearchSpace = field(
        default_factory=LossBalanceSearchSpace
    )
    loss_auxiliary: LossAuxiliarySearchSpace = field(
        default_factory=LossAuxiliarySearchSpace
    )
    regularization: RegularizationSearchSpace = field(
        default_factory=RegularizationSearchSpace
    )
    head_weights: HeadWeightsSearchSpace = field(
        default_factory=HeadWeightsSearchSpace
    )
    mtl_joint: MtlJointSearchSpace = field(
        default_factory=MtlJointSearchSpace
    )
    hierarchy: HierarchySearchSpace = field(
        default_factory=HierarchySearchSpace
    )

    def validate(self) -> None:
        self.loss_balance.validate()
        self.loss_auxiliary.validate()
        self.regularization.validate()
        self.head_weights.validate()
        self.mtl_joint.validate()
        self.hierarchy.validate()
