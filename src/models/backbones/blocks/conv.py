'''Convolution blocks for UNet families'''

# third-part imports
import torch
import torch.nn

class DoubleConv(torch.nn.Module):
    '''Two stacked Conv-Norm-ReLU units with optional dropout.'''

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            **kwargs
        ):
        '''Init double-conv block with configurable normalization.'''
        super().__init__()

        # unpack keyword arguments
        norm = kwargs.get('norm', 'gn') # default group norm
        gn_groups = kwargs.get('gn_groups', 8) # if group norm is used
        p_drop = kwargs.get('p_drop', 0.05) # drop out rate
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            self._get_norm(norm, out_ch, gn_groups),
            torch.nn.ReLU(inplace=False),
            torch.nn.Dropout2d(p_drop) if p_drop > 0 else torch.nn.Identity(),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            self._get_norm(norm, out_ch, gn_groups),
            torch.nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Apply two Conv-Norm-ReLU stages (with optional dropout).'''
        return self.block(x)

    @staticmethod
    def _get_norm(
            kind: str | None,
            num_channels: int,
            gn_groups: int = 8
        ) -> torch.nn.Module:
        '''Return the requested normalization layer or identity.'''

        # no normalization
        if kind is None or kind.lower() == 'none':
            return torch.nn.Identity() # Indentity layer
        # availabel methods: batchNorm, GroupNorm, and LayerNorm
        kind = kind.lower()
        if kind == 'bn':
            return torch.nn.BatchNorm2d(num_channels)
        if kind == 'gn':
            g = min(gn_groups, num_channels)
            while num_channels % g != 0 and g > 1: # fall back to (1, ...)
                g -= 1
            return torch.nn.GroupNorm(g, num_channels, eps=1e-4, affine=True)
        if kind == 'ln':
            return torch.nn.GroupNorm(1, num_channels, affine=True)
        raise ValueError(f'Unknown norm type: {kind}')

class Downsample(torch.nn.Module):
    '''Max-pool downsampling followed by a double-conv block.'''

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            **kwargs
        ):
        '''Initialize a pooling-plus-double-conv downsampling stage.'''

        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, **kwargs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Downsample by 2x and refine with convolutions.'''

        return self.block(x)

class Upsample(torch.nn.Module):
    '''Bilinear upsampling, skip concatenation, and double-conv.'''

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            **kwargs
        ):
        '''Initialize the upsample-and-fuse stage with a double-conv.'''

        super().__init__()
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.conv = DoubleConv(in_ch, out_ch, **kwargs)

    def forward(
            self,
            x: torch.Tensor,
            skip: torch.Tensor
        ) -> torch.Tensor:
        '''Upsample, concatenate the skip feature, and refine.'''

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1) # concatenate skip connection here
        x = x.contiguous()
        return self.conv(x)
