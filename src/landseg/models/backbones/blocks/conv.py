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

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
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

        # unpack keyword arguments
        norm = kwargs.get('norm', 'gn') # default group norm
        gn_groups = kwargs.get('gn_groups', 8) # if group norm is used
        p_drop = kwargs.get('p_drop', 0.05) # drop out rate
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
    def _norm(mode: str | None, ch: int, gn_groups: int=8) -> torch.nn.Module:
        '''Return the requested normalization layer or identity.'''

        # no normalization
        if mode is None or mode.lower() == 'none':
            return torch.nn.Identity() # Indentity layer
        # availabel methods: batchNorm, GroupNorm, and LayerNorm
        mode = mode.lower()
        if mode == 'bn':
            return torch.nn.BatchNorm2d(ch)
        if mode == 'gn':
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

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
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
            DoubleConv(in_ch, out_ch, **kwargs)
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

    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        '''
        Bilinear upsample, skip concatenation, and DoubleConv refinement.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            **kwargs: Additional options pass to `DoubleConv`.
        '''

        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        '''Upsample, concatenate the skip feature, and refine.'''

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1) # concatenate skip connection here
        x = x.contiguous()
        return self.conv(x)
