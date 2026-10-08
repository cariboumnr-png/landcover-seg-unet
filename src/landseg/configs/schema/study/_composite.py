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
Study configuration section.
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.study.architecture as architect
import landseg.configs.schema.study.objectives as objectives
import landseg.configs.schema.study.optimization as optimization

# alias
field = dataclasses.field




@dataclasses.dataclass
class StudyConfig(base.BaseConfigSection):
    base: optimization.BaseSearchSpace = field(
        default_factory=optimization.BaseSearchSpace
    )
    optimizer: optimization.OptimizerSearchSpace = field(
        default_factory=optimization.OptimizerSearchSpace
    )
    throughput: optimization.ThroughputSearchSpace = field(
        default_factory=optimization.ThroughputSearchSpace
    )
    data_geometry: architect.DataGeometrySearchSpace = field(
        default_factory=architect.DataGeometrySearchSpace
    )
    context_window: architect.ContextWindowSearchSpace = field(
        default_factory=architect.ContextWindowSearchSpace
    )
    architecture: architect.ArchitectureSearchSpace = field(
        default_factory=architect.ArchitectureSearchSpace
    )
    bottleneck: architect.BottleneckSearchSpace = field(
        default_factory=architect.BottleneckSearchSpace
    )
    conditioning: architect.ConditioningSearchSpace = field(
        default_factory=architect.ConditioningSearchSpace
    )
    loss_balance: objectives.LossBalanceSearchSpace = field(
        default_factory=objectives.LossBalanceSearchSpace
    )
    loss_auxiliary: objectives.LossAuxiliarySearchSpace = field(
        default_factory=objectives.LossAuxiliarySearchSpace
    )
    regularization: objectives.RegularizationSearchSpace = field(
        default_factory=objectives.RegularizationSearchSpace
    )
    head_weights: objectives.HeadWeightsSearchSpace = field(
        default_factory=objectives.HeadWeightsSearchSpace
    )
    mtl_joint: objectives.MtlJointSearchSpace = field(
        default_factory=objectives.MtlJointSearchSpace
    )
    hierarchy: objectives.HierarchySearchSpace = field(
        default_factory=objectives.HierarchySearchSpace
    )
