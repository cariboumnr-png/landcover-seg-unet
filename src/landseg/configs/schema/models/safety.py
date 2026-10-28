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

'''Model numeric safety configuration.'''

# standard imports
import dataclasses
# local imports
import landseg.configs.schema.base as base


@dataclasses.dataclass
class NumericSafety(base.BaseConfigSection):
    '''Model numeric safety configuration.'''
    enable_clamp: bool = True
    clamp_range: tuple[float, float] = (1e-4, 1e4)

    def validate(self):
        self.val.require_sequence(self.clamp_range, 2)
        lo, hi = self.clamp_range
        self.val.require_numeric_min(lo, 0.0, float)
        self.val.require_numeric_min(hi, 0.0, float)
        if lo >= hi:
            raise ValueError(f'Invalid clamp {self.clamp_range}; low <= high')
