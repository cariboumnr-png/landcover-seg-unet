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
Data configs section
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base
import landseg.configs.schema.data as data

# alias
field = dataclasses.field


@dataclasses.dataclass
class DataConfig(base.BaseConfigSection):
    world_grid: data.WorldGridConfig = field(
        default_factory=data.WorldGridConfig
    )
    harmonization: data.DataHarmonizationConfig = field(
        default_factory=data.DataHarmonizationConfig
    )
    ingestion: data.DataIngestionConfig = field(
        default_factory=data.DataIngestionConfig
    )
    preparation: data.DataPreparationConfig = field(
        default_factory=data.DataPreparationConfig
    )
    specification: data.DataSpecificationConfig = field(
        default_factory=data.DataSpecificationConfig
    )

    def validate(self):
        self.world_grid.validate()
        self.harmonization.validate()
        self.ingestion.validate()
        self.preparation.validate()
        self.specification.validate()
