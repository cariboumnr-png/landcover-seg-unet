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

'''Base classes for loss component.'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class PrimitiveLoss(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Base class for loss primitives.'''
    @abc.abstractmethod
    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            *,
            masks: dict[float, torch.Tensor] | None,
        ) -> torch.Tensor:
        '''Forward.'''
