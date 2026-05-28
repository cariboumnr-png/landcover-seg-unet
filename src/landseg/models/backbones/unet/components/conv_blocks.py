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
Convolution blocks for UNet-like encoder-decoder models.

**Overview**
This module provides the common building blocks used across UNet-likes:
- DoubleConv: two Conv2d → Norm → ReLU stages (optional dropout).
- Downsample: 2x max-pool followed by a DoubleConv.
- Upsample: bilinear upsample, skip concatenation, then DoubleConv.

**Expected tensor shapes**
- Inputs/outputs are 4D tensors: (N, C, H, W).
- Spatial size halves with Downsample and doubles with Upsample.
- Channel counts are determined by in_ch/out_ch passed to the blocks.

**Notes**
- GroupNorm with 1 group behaves like LayerNorm over channels for 2D features.
- Normalization can be disabled with norm=None (Identity).
'''

# third-part imports
import torch
import torch.nn
# local imports
import landseg.models.backbones.unet.components as components

class DoubleConv(torch.nn.Module):
    '''
    Two stacked Conv-Norm-ReLU units with optional dropout.

    **Architecture**\n
    (Conv2d → Norm → ReLU → Dropout2d?) x 1 → Conv2d → Norm → ReLU

    **Notes**
    - Returns a torch.Tensor as feature map of shape (N, out_ch, H, W).
    - Convs use kernel_size=3, padding=1, bias=False.
    - ReLU configured with inplace=False to be conservative with autograd.
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        params: components.ConvolutionParameters | None,
    ):
        '''
        Initiate a double-conv block with configurable normalization.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            **kwargs: Additional options pass to convolutional blocks.
                - `norm` (str | None): normalization kind; one of
                    {'bn', 'gn', 'ln', None}. Default: 'gn'.
                - `gn_groups` (int): number of groups when `norm='gn'`.
                    Automatically reduced to a divisor of channels.
                    Default: 8.
                - `p_drop` (float): dropout probability used in
                    `DoubleConvBlock`. Default: 0.05.
        '''
        super().__init__()

        # resolve parameters
        if params is None:
            norm = None
            gn_groups = None
            p_drop = 0.0
        else:
            norm = params.norm
            gn_groups = params.gn_groups
            p_drop = params.p_drop

        # double convolutions
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            self._norm(norm, out_ch, gn_groups),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout2d(p_drop) if p_drop > 0 else torch.nn.Identity(),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            self._norm(norm, out_ch, gn_groups),
            torch.nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply two Conv-Norm-ReLU stages (with optional dropout).'''
        return self.block(x)

    @staticmethod
    def _norm(mode: str | None, ch: int, gn_groups: int | None) -> torch.nn.Module:
        '''Return the requested normalization layer or identity.'''

        # no normalization
        if mode is None or mode.lower() == 'none':
            return torch.nn.Identity() # Indentity layer
        # availabel methods: batchNorm, GroupNorm, and LayerNorm
        mode = mode.lower()
        if mode == 'bn':
            return torch.nn.BatchNorm2d(ch)
        if mode == 'gn':
            assert gn_groups is not None, 'gn_groups must be specified for group norm'
            g = min(gn_groups, ch)
            while ch % g != 0 and g > 1: # fall back to (1, ...)
                g -= 1
            return torch.nn.GroupNorm(g, ch, eps=1e-4, affine=True)
        if mode == 'ln':
            return torch.nn.GroupNorm(1, ch, affine=True)
        raise ValueError(f'Unknown norm type: {mode}')

class Downsample(torch.nn.Module):
    '''
    Max-pool downsampling followed by a DoubleConv.

    **Architecture**\n
    MaxPool2d(stride=2) → DoubleConv(in_ch, out_ch, **kwargs)
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        params: components.ConvolutionParameters,
    ):
        '''
        Initialize a pooling-plus-double-conv downsampling stage.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            **kwargs: Additional options pass to `DoubleConv`.
        '''

        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Downsample by 2x and refine with convolutions.'''

        return self.block(x)

class Upsample(torch.nn.Module):
    '''
    Bilinear upsample, skip concatenation, and DoubleConv refinement.

    **Architecture**\n
    Upsample(scale_factor=2) → Concat([x_up, skip], dim=1) → DoubleConv
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        params: components.ConvolutionParameters
    ):
        '''
        Bilinear upsample, skip concatenation, and DoubleConv refinement.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            params: Convolution parameters.
        '''

        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.conv = DoubleConv(in_ch, out_ch, params)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        '''Upsample, concatenate the skip feature, and refine.'''

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1) # concatenate skip connection here
        x = x.contiguous()
        return self.conv(x)
