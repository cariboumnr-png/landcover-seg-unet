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

'''Base classes for architecture backbones.'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.backbones.unet.components as components

class UNetEncoders(torch.nn.Module):
    '''
    Contract for any feature extractor used by MultiHeadModel.
    Implementations must produce a feature map with a known channel
    width that matches the heads' expected input (e.g., base_ch).
    '''

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        params: components.ConvolutionParameters,
    ) -> None:
        super().__init__()

        ch = base_ch # alias base_ch -> ch
        # initial convolution block with no norm nor drop outs
        self.inc = components.DoubleConv(in_ch, base_ch, params)
        # downsampling path (encoder) with 4 levels
        self.downs = torch.nn.ModuleList([
            components.Downsample(ch,   ch*2,  params),
            components.Downsample(ch*2, ch*4,  params),
            components.Downsample(ch*4, ch*8,  params),
            components.Downsample(ch*8, ch*16, params),
        ])

    def __len__(self) -> int:
        '''Return the number of encoder levels.'''
        return len(self.downs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''Return 5-level encoder features.'''

        x1 = self.inc(x)                # H     in_ch ->  b
        x2 = self.downs[0](x1)          # H/2   b     ->  2b
        x3 = self.downs[1](x2)          # H/4   2b    ->  4b
        x4 = self.downs[2](x3)          # H/8   4b    ->  8b
        x5 = self.downs[3](x4)          # H/16  8b    ->  16b

        return x1, x2, x3, x4, x5
