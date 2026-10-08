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
import landseg.configs.schema.study.architecture as architecture_sec
import landseg.configs.schema.study.objectives as objectives_sec
import landseg.configs.schema.study.optimization as optimization_sec

# ----- aliases
field = dataclasses.field
BaseSearch = optimization_sec.BaseSearchSpace
OptimizerSearch = optimization_sec.OptimizerSearchSpace
ThroughputSearch = optimization_sec.ThroughputSearchSpace
DataGeometrySearch = architecture_sec.DataGeometrySearchSpace
ContextWindowSearch = architecture_sec.ContextWindowSearchSpace
ArchitectureSearch = architecture_sec.ArchitectureSearchSpace
BottleneckSearch = architecture_sec.BottleneckSearchSpace
ConditioningSearch = architecture_sec.ConditioningSearchSpace
LossBalanceSearch = objectives_sec.LossBalanceSearchSpace
LossAuxiliarySearch = objectives_sec.LossAuxiliarySearchSpace
RegularizationSearch = objectives_sec.RegularizationSearchSpace
HeadWeightsSearch = objectives_sec.HeadWeightsSearchSpace
MtlJointSearch = objectives_sec.MtlJointSearchSpace
HierarchySearch = objectives_sec.HierarchySearchSpace


@dataclasses.dataclass
class StudyConfig(base.BaseConfigSection):
    base: BaseSearch = field(default_factory=BaseSearch)
    optimizer: OptimizerSearch = field(default_factory=OptimizerSearch)
    throughput: ThroughputSearch = field(default_factory=ThroughputSearch)
    data_geometry: DataGeometrySearch = field(default_factory=DataGeometrySearch)
    context_window: ContextWindowSearch = field(default_factory=ContextWindowSearch)
    architecture: ArchitectureSearch = field(default_factory=ArchitectureSearch)
    bottleneck: BottleneckSearch = field(default_factory=BottleneckSearch)
    conditioning: ConditioningSearch = field(default_factory=ConditioningSearch)
    loss_balance: LossBalanceSearch = field(default_factory=LossBalanceSearch)
    loss_auxiliary: LossAuxiliarySearch = field(default_factory=LossAuxiliarySearch)
    regularization: RegularizationSearch = field(default_factory=RegularizationSearch)
    head_weights: HeadWeightsSearch = field(default_factory=HeadWeightsSearch)
    mtl_joint: MtlJointSearch = field(default_factory=MtlJointSearch)
    hierarchy: HierarchySearch = field(default_factory=HierarchySearch)
