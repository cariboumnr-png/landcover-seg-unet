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
Study schema.
'''

# standard imports
import dataclasses

# alias
field = dataclasses.field

# --------------------------------STUDY CONFIGS--------------------------------
@dataclasses.dataclass
class _BaseObjectives:
    learning_rate: tuple[float, float] = (1e-5, 1e-1)   # low, high
    weight_decay: tuple[float, float] = (1e-6, 1e-2)    # low, high
    patch_size: tuple[int, int, int] = (64, 128, 64)    # low, high, step
    batch_size: tuple[int, int, int] = (16, 64, 16)     # low, high, step

@dataclasses.dataclass
class StudyConfig:
    base: _BaseObjectives = field(default_factory=_BaseObjectives)
