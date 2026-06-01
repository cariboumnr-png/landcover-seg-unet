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

'''
Numerical stability utilities for autocast and tensor clamping.

This module provides lightweight utilities for controlling numerical
precision via mixed-precision autocast and value clamping to prevent
gradient explosion or vanishing during training and inference.
'''

# third-party imports
import torch

class NumericSafety:
    '''Autocast and clamping utilities for numerical stability.'''

    def __init__(
        self,
        *,
        enable_clamp: bool,
        clamp_range: tuple[float, float],
        device: str
    ):
        '''Configure clamping behavior and bounds.'''

        self.enable_clamp = enable_clamp
        self.clamp_range = clamp_range
        self.device = device

    def autocast_context(
        self,
        enable: bool = True,
        dtype: torch.dtype = torch.float16
    ) -> torch.autocast:
        '''Create an AMP autocast context for the current device.'''

        return torch.autocast(self.device, dtype, enable)

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        '''Clamp tensor values to a safe numeric range.'''

        if not self.enable_clamp:
            return x
        mmin, mmax = self.clamp_range
        return torch.clamp(x, min=mmin, max=mmax)
