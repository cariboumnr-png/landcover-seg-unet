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

'''Base interface for multihead model.'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn

class BaseMultiheadModel(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Minimally required class methods.'''

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        '''Multihead output as a dict of tensors'''

    @abc.abstractmethod
    def set_active_heads(self, active_heads: list[str] | None) -> None:
        '''Heads to be actively trained.'''

    @abc.abstractmethod
    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        '''Heads with no gradient updates.'''

    @abc.abstractmethod
    def reset_heads(self) -> None:
        '''Reset heads state.'''

    @abc.abstractmethod
    def set_logit_adjust_enabled(self, enabled: bool) -> None:
        '''Enable/disable logit adjustment.'''

    @abc.abstractmethod
    def set_logit_adjust_alpha(self, alpha: float) -> None:
        '''Set global logit adjustment scalar.'''
