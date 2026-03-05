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

'''Head specifications.'''

# standard imports
import dataclasses

@dataclasses.dataclass
class Spec:
    '''Specifications for a training head.'''
    name: str
    count: list[int]
    loss_alpha: list[float]
    parent_head: str | None
    parent_cls: int | None # 1-based
    weight: float
    exclude_cls: tuple[int, ...]

    def set_exclude(self, indices: tuple[int, ...]) -> None:
        '''Curriculum fills this during training; validates range.'''
        bad = [i for i in indices if i < 1 or i > self.num_classes]
        if bad:
            raise ValueError(f'Invalid indices to exclude: {bad}')
        self.exclude_cls = tuple(sorted(set(indices)))

    @property
    def num_classes(self) -> int:
        '''Number of classes in this head.'''
        return len(self.count)
