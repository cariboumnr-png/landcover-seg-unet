# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
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
Example UNet variants using transformer bottleneck.

This module demonstrates how to create UNet variants that use the
transformer-based bottleneck module as a drop-in replacement for the
standard convolutional bottleneck.

**Variants Provided**
- `UNetTransformer`: Pure transformer bottleneck (best accuracy potential)
- `UNetHybrid`: Mixed conv + transformer (best efficiency)
- `UNetLightTransformer`: Lightweight transformer (lower memory footprint)
'''

# third-party imports
import torch
import torch.nn

# local imports
import landseg.models.backbones.unet as unet
import landseg.models.backbones.unet.components as components

class UNetTransformer(unet.UNetBackbone):
    '''
    UNet with pure transformer bottleneck.

    Replaces the standard convolutional bottleneck with a stack of
    Vision Transformer blocks. This variant maximizes the use of
    long-range dependencies at the bottleneck level.

    **Use when:**
    - You have sufficient GPU memory
    - Segmentation targets have large spatial patterns
    - You want maximum model capacity at the bottleneck

    **Architecture:**
    - Standard UNet encoder-decoder
    - Transformer bottleneck (4 blocks, 8 heads)
    - Best for capturing global context
    '''

    def __init__(self, in_ch: int, base_ch: int, **kwargs):
        super().__init__()
        self._out_channels = base_ch
        ch = base_ch

        # Initial convolution
        self.inc = components.DoubleConv(in_ch, ch, norm=None, p_drop=0.0)

        # Encoder (downsampling path)
        self.downs = torch.nn.ModuleList([
            components.Downsample(ch, ch*2, **kwargs.get('downs', {})),
            components.Downsample(ch*2, ch*4, **kwargs.get('downs', {})),
            components.Downsample(ch*4, ch*8, **kwargs.get('downs', {})),
            components.Downsample(ch*8, ch*16, **kwargs.get('downs', {})),
        ])

        # Transformer bottleneck
        transformer_config = kwargs.get('bottleneck', {})
        self.bottleneck = components.TransformerBottleneck(
            in_channels=ch * 16,
            num_blocks=transformer_config.get('num_blocks', 4),
            num_heads=transformer_config.get('num_heads', 8),
            mlp_ratio=transformer_config.get('mlp_ratio', 4.0),
            dropout=transformer_config.get('dropout', 0.1),
            attn_dropout=transformer_config.get('attn_dropout', 0.0),
        )

        # Decoder (upsampling path)
        self.ups = torch.nn.ModuleList([
            components.Upsample(ch*16 + ch*8, ch*8, **kwargs.get('ups', {})),
            components.Upsample(ch*8 + ch*4, ch*4, **kwargs.get('ups', {})),
            components.Upsample(ch*4 + ch*2, ch*2, **kwargs.get('ups', {})),
            components.Upsample(ch*2 + ch, ch, **kwargs.get('ups', {}))
        ])

        # Kaiming weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    @property
    def bottleneck_ch(self) -> int:
        return self._out_channels * 16

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_divisor(self) -> int:
        return 2 ** len(self.downs)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.inc(x)
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)
        xb = self.bottleneck(x5)
        return x1, x2, x3, x4, xb

    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x1, x2, x3, x4, xb = xs
        x = self.ups[0](xb, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, xb = self.encode(x)
        return self.decode((x1, x2, x3, x4, xb))


class UNetHybrid(unet.UNetBackbone):
    '''
    UNet with hybrid (conv + transformer) bottleneck.

    Combines 1-2 convolutional blocks with transformer blocks for an
    efficient middle ground. Recommended for most production scenarios.

    **Use when:**
    - You need a balance between accuracy and speed
    - GPU memory is a concern
    - You want better computational efficiency than pure transformer

    **Architecture:**
    - Standard UNet encoder-decoder
    - Hybrid bottleneck (1 conv + 2 transformer blocks)
    - Good accuracy/speed tradeoff
    '''

    def __init__(self, in_ch: int, base_ch: int, **kwargs):
        super().__init__()
        self._out_channels = base_ch
        ch = base_ch

        self.inc = components.DoubleConv(in_ch, ch, norm=None, p_drop=0.0)

        self.downs = torch.nn.ModuleList([
            components.Downsample(ch, ch*2, **kwargs.get('downs', {})),
            components.Downsample(ch*2, ch*4, **kwargs.get('downs', {})),
            components.Downsample(ch*4, ch*8, **kwargs.get('downs', {})),
            components.Downsample(ch*8, ch*16, **kwargs.get('downs', {})),
        ])

        # Hybrid bottleneck
        bottleneck_config = kwargs.get('bottleneck', {})
        self.bottleneck = components.HybridBottleneck(
            in_channels=ch * 16,
            num_conv_blocks=bottleneck_config.get('num_conv_blocks', 1),
            num_transformer_blocks=bottleneck_config.get('num_transformer_blocks', 2),
            num_heads=bottleneck_config.get('num_heads', 8),
            mlp_ratio=bottleneck_config.get('mlp_ratio', 2.0),
            dropout=bottleneck_config.get('dropout', 0.1),
            attn_dropout=bottleneck_config.get('attn_dropout', 0.0),
            norm=bottleneck_config.get('norm', None)
        )

        self.ups = torch.nn.ModuleList([
            components.Upsample(ch*16 + ch*8, ch*8, **kwargs.get('ups', {})),
            components.Upsample(ch*8 + ch*4, ch*4, **kwargs.get('ups', {})),
            components.Upsample(ch*4 + ch*2, ch*2, **kwargs.get('ups', {})),
            components.Upsample(ch*2 + ch, ch, **kwargs.get('ups', {}))
        ])

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    @property
    def bottleneck_ch(self) -> int:
        return self._out_channels * 16

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_divisor(self) -> int:
        return 2 ** len(self.downs)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.inc(x)
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)
        xb = self.bottleneck(x5)
        return x1, x2, x3, x4, xb

    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x1, x2, x3, x4, xb = xs
        x = self.ups[0](xb, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, xb = self.encode(x)
        return self.decode((x1, x2, x3, x4, xb))


class UNetLightTransformer(unet.UNetBackbone):
    '''
    Lightweight UNet with minimal transformer bottleneck.

    Uses only 2 transformer blocks with 4 attention heads for minimal
    memory overhead while capturing some long-range dependencies.

    **Use when:**
    - Memory is very constrained
    - You want to experiment with transformers at lower cost
    - Speed is critical

    **Architecture:**
    - Standard UNet encoder-decoder
    - Lightweight transformer (2 blocks, 4 heads)
    - Minimal memory overhead vs standard UNet
    '''

    def __init__(self, in_ch: int, base_ch: int, **kwargs):
        super().__init__()
        self._out_channels = base_ch
        ch = base_ch

        self.inc = components.DoubleConv(in_ch, ch, norm=None, p_drop=0.0)

        self.downs = torch.nn.ModuleList([
            components.Downsample(ch, ch*2, **kwargs.get('downs', {})),
            components.Downsample(ch*2, ch*4, **kwargs.get('downs', {})),
            components.Downsample(ch*4, ch*8, **kwargs.get('downs', {})),
            components.Downsample(ch*8, ch*16, **kwargs.get('downs', {})),
        ])

        # Lightweight transformer
        self.bottleneck = components.TransformerBottleneck(
            in_channels=ch * 16,
            num_blocks=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.05,
            attn_dropout=0.0,
        )

        self.ups = torch.nn.ModuleList([
            components.Upsample(ch*16 + ch*8, ch*8, **kwargs.get('ups', {})),
            components.Upsample(ch*8 + ch*4, ch*4, **kwargs.get('ups', {})),
            components.Upsample(ch*4 + ch*2, ch*2, **kwargs.get('ups', {})),
            components.Upsample(ch*2 + ch, ch, **kwargs.get('ups', {}))
        ])

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    @property
    def bottleneck_ch(self) -> int:
        return self._out_channels * 16

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def spatial_divisor(self) -> int:
        return 2 ** len(self.downs)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x1 = self.inc(x)
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)
        xb = self.bottleneck(x5)
        return x1, x2, x3, x4, xb

    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x1, x2, x3, x4, xb = xs
        x = self.ups[0](xb, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4, xb = self.encode(x)
        return self.decode((x1, x2, x3, x4, xb))
