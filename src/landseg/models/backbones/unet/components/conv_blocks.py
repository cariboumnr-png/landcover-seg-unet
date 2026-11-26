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

'''
Convolution blocks for UNet-like encoder-decoder models.

Provides the standard convolutional building blocks shared across UNet
backbones, operating on 4D `(N, C, H, W)` feature tensors.

Notes:
    - `GroupNorm` with 1 group behaves like `LayerNorm` over channels.
    - Normalization can be disabled by passing `norm=None`.

Public APIs:
    - `DoubleConv`: Two stacked Conv-Norm-ReLU stages with dropout.
    - `Downsample`: Max-pool downsampling followed by `DoubleConv`.
    - `Upsample`: Bilinear upsample, skip concat, and `DoubleConv`.
'''

# third-part imports
import torch
import torch.nn
# local imports
import landseg.models.backbones.unet.components as components

class DoubleConv(torch.nn.Module):
    '''
    Two stacked Conv-Norm-ReLU units with optional dropout.

    Architecture:
        `Conv2d -> Norm -> ReLU -> Dropout2d? -> Conv2d -> Norm -> ReLU`

    Notes:
        - Produces a `torch.Tensor` of shape `(N, out_ch, H, W)`.
        - Convolutions use `kernel_size=3`, `padding=1`, `bias=False`.
        - `ReLU` is configured with `inplace=False` for autograd safety.
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        params: components.ConvolutionParameters | None,
    ):
        '''
        Initialize a double-convolution block with optional norm.

        Args:
            in_ch:
                Number of input channels.
            out_ch:
                Number of output channels.
            params:
                Optional `ConvolutionParameters` specifying
                normalization mode, group count, and dropout rate. When
                `None`, normalization and dropout are omitted.
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
        '''
        Apply two Conv-Norm-ReLU stages with optional dropout.

        Args:
            x:
                Input feature tensor of shape `(B, in_ch, H, W)`.

        Returns:
            torch.Tensor:
                Output feature tensor of shape `(B, out_ch, H, W)`.
        '''
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
    Max-pool downsampling stage followed by `DoubleConv`.

    Applies `MaxPool2d(2)` followed by a `DoubleConv` block.
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
            in_ch:
                Number of input channels.
            out_ch:
                Number of output channels.
            params:
                `ConvolutionParameters` forwarded to `DoubleConv`.
        '''
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, params)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Downsample input by 2x and refine with `DoubleConv`.

        Args:
            x:
                Input feature tensor of shape `(B, in_ch, H, W)`.

        Returns:
            torch.Tensor:
                Downsampled feature tensor of shape
                `(B, out_ch, H // 2, W // 2)`.
        '''
        return self.block(x)

class Upsample(torch.nn.Module):
    '''
    Bilinear upsample, skip concatenation, and `DoubleConv` refinement.

    Upsamples input by a factor of 2, concatenates the matching encoder
    skip tensor along channels, and applies `DoubleConv`.
    '''

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        params: components.ConvolutionParameters
    ):
        '''
        Initialize upsample, skip concatenation, and `DoubleConv`.

        Args:
            in_ch:
                Number of concatenated input channels.
            out_ch:
                Number of output channels.
            params:
                `ConvolutionParameters` forwarded to `DoubleConv`.
        '''
        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.conv = DoubleConv(in_ch, out_ch, params)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        '''
        Upsample feature map, concatenate skip, and refine.

        Args:
            x:
                Coarse feature tensor to upsample.
            skip:
                Skip-connection tensor at the target resolution.

        Returns:
            torch.Tensor:
                Refined feature tensor with `out_ch` channels.
        '''
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1) # concatenate skip connection here
        x = x.contiguous()
        return self.conv(x)
