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
OptimizationSearch = optimization_sec.OptimizationSearchConfig
ArchitectureSearch = architecture_sec.ArchitectureSearchConfig
ObjectivesSearch = objectives_sec.ObjectivesSearchConfig


@dataclasses.dataclass
class StudyConfig(base.BaseConfigSection):
    optimization: OptimizationSearch = field(default_factory=OptimizationSearch)
    architecture: ArchitectureSearch = field(default_factory=ArchitectureSearch)
    objectives: ObjectivesSearch = field(default_factory=ObjectivesSearch)

    def validate(self) -> None:
        self.optimization.validate()
        self.architecture.validate()
        self.objectives.validate()
