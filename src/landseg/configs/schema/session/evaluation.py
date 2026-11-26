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

'''Session model evaluation configuration.'''

# standard imports
import dataclasses
import typing
# local imports
import landseg.configs.schema.base as base


@dataclasses.dataclass
class EvaluationConfig(base.BaseConfigSection):
    '''Configuration for model evaluation runs.'''
    checkpoint: str | None = None
    split: str = 'test'
    export_previews: bool = False

    @property
    def valid_split(self) -> typing.Literal['val', 'test']:
        '''Return validated evaluation split identifier.'''
        if self.split not in ('val', 'test'):
            raise base.ConfigValidationError(f'Invalid split: {self.split}')
        return typing.cast(typing.Literal['val', 'test'], self.split)

    def validate(self) -> None:
        '''Validate that evaluation split is `'val'` or `'test'`.'''
        self.val.require_value_from(self.split, ['val', 'test'], str)
