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
Data harmonziation configs
'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base

# alias
field = dataclasses.field


CONT = ['cubic', 'bilinear']
CATE = ['nearest']


@dataclasses.dataclass
class DataHarmonizationConfig(base.BaseConfigSection):
    dataset_manifest: str = ''
    resampling_continuous: str = 'bilinear'
    resampling_categorical: str = 'nearest'
    output_dpath: str = 'experiment/artifacts/harmonized_data'

    def validate(self) -> None:
        self.require_file(self.dataset_manifest, 'source dataset manifest')
        self.require_attr_type_range('resampling_continuous', str, CONT)
        self.require_attr_type_range('resampling_categorical', str, CATE)
