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
Bottleneck modules for UNet encoder-decoder architectures.

Provides the abstract bottleneck base contract alongside standard
convolutional, Vision Transformer, and hybrid convolution-transformer
bottleneck implementations.

Design Points:
    - Preserves input/output channel width (`in_ch == out_ch`).
    - Preserves spatial dimensions without downsampling.
    - Uses residual connections in transformer blocks for gradient flow.

Public APIs:
    - `BaseBottleneck`: Abstract base class for UNet bottlenecks.
    - `HybridBottleneck`: Combined convolutional-transformer bottleneck.
    - `TransformerBottleneck`: Vision Transformer bottleneck stack.
    - `UNetBottleneck`: Standard convolutional UNet bottleneck block.
'''

# standard imports
import abc
# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.backbones.unet.components as components

class BaseBottleneck(torch.nn.Module):
    '''
    Base class for UNet bottlenecks.

    Defines the interface and common functionality for bottleneck
    modules used in UNet architectures. Subclasses must implement
    `forward` to process the input feature map and produce an output of
    identical shape.
    '''

    def __init__(self, in_ch: int):
        '''
        Initialize base bottleneck state.

        Args:
            in_ch:
                Number of input and output feature channels.
        '''
        super().__init__()
        self.in_ch = in_ch

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process input feature map through the bottleneck.

        Args:
            x:
                Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor:
                Output tensor of shape `(B, C, H, W)`.

        Raises:
            NotImplementedError:
                If the subclass does not implement `forward`.
        '''
        raise NotImplementedError

class UNetBottleneck(BaseBottleneck):
    '''
    Standard convolutional bottleneck for UNet.

    Implements a `DoubleConv` block at the UNet bottleneck with
    `in_ch == out_ch` and no spatial downsampling.
    '''

    def __init__(
        self,
        in_ch: int,
        params: components.ConvolutionParameters
    ):
        '''
        Initialize convolutional bottleneck block.

        Args:
            in_ch:
                Number of input and output feature channels.
            params:
                `ConvolutionParameters` configuring normalization and
                dropout.
        '''
        super().__init__(in_ch)
        self.double_conv = components.DoubleConv(in_ch, in_ch, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply double-convolution bottleneck transformation.

        Args:
            x:
                Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor:
                Output tensor of shape `(B, C, H, W)`.
        '''
        return self.double_conv(x)

class TransformerBottleneck(BaseBottleneck):
    '''
    Transformer-based bottleneck for U-Net.

    Replaces the standard convolutional bottleneck with a stack of
    transformer blocks, enabling long-range dependency modeling while
    preserving spatial structure and tensor shape.
    '''

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        config: components.BottleneckConfig,
    ):
        '''
        Initialize transformer bottleneck layers.

        Args:
            in_channels:
                Number of input and output channels.
            spatial_size:
                Expected spatial height/width at the bottleneck.
            config:
                `BottleneckConfig` providing transformer block count and
                hyperparameters.
        '''
        super().__init__(in_channels)

        assert config.num_transformer_blocks
        assert config.transformer_params
        self.in_ch = in_channels

        self.pos_embed = _PositionalEmbedding(in_channels, spatial_size)

        self.blocks = torch.nn.ModuleList([
            _TransformerBlock(
                in_channels,
                num_heads=config.transformer_params.num_heads,
                mlp_ratio=config.transformer_params.mlp_ratio,
                dropout=config.transformer_params.dropout,
                attn_dropout=config.transformer_params.attn_dropout,
            )
            for _ in range(config.num_transformer_blocks)
        ])

        self.norm_out = torch.nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply transformer bottleneck.

        Args:
            x:
                Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor:
                Output tensor of shape `(B, C, H, W)`.
        '''
        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)

        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm_out(x_flat)
        x = x_flat.transpose(1, 2).reshape(b, c, h, w)

        return x

class HybridBottleneck(BaseBottleneck):
    '''
    Hybrid bottleneck combining convolution and transformer blocks.

    Provides a middle ground between purely convolutional and purely
    transformer bottlenecks, combining local inductive bias with
    long-range self-attention.
    '''

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        config: components.BottleneckConfig,
    ):
        '''
        Initialize hybrid convolution-transformer bottleneck.

        Args:
            in_channels:
                Number of input and output channels.
            spatial_size:
                Expected spatial height/width at the bottleneck.
            config:
                `BottleneckConfig` providing convolution and transformer
                counts and hyperparameters.
        '''
        super().__init__(in_channels)
        self.in_ch = in_channels

        assert config.num_conv_blocks
        assert config.conv_params
        assert config.num_transformer_blocks
        assert config.transformer_params

        self.conv_blocks = torch.nn.ModuleList([
            components.DoubleConv(in_channels, in_channels, config.conv_params)
            for _ in range(config.num_conv_blocks)
        ])

        self.pos_embed = _PositionalEmbedding(in_channels, spatial_size)

        self.transformer_blocks = torch.nn.ModuleList([
            _TransformerBlock(
                in_channels,
                num_heads=config.transformer_params.num_heads,
                mlp_ratio=config.transformer_params.mlp_ratio,
                dropout=config.transformer_params.dropout,
                attn_dropout=config.transformer_params.attn_dropout,
            )
            for _ in range(config.num_transformer_blocks)
        ])

        self.norm_out = torch.nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply hybrid bottleneck (conv + transformer).

        Args:
            x:
                Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor:
                Output tensor of shape `(B, C, H, W)`.
        '''
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        x = self.pos_embed(x)

        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.norm_out(x_flat)
        x = x_flat.transpose(1, 2).reshape(b, c, h, w)

        return x

class _PositionalEmbedding(torch.nn.Module):
    '''
    Learnable absolute positional embeddings for 2D feature maps.

    Applies learnable position-wise embeddings initialized with values
    that encourage stable convergence. Compatible with varying spatial
    dimensions through reshape-based broadcasting.
    '''

    def __init__(self, channels: int, spatial_size: int):
        '''
        Initialize learnable 2D positional embedding parameter.

        Args:
            channels:
                Number of feature channels.
            spatial_size:
                Nominal spatial height and width for initialization.
        '''
        super().__init__()
        self.channels = channels
        self.register_parameter(
            'pos_embed',
            torch.nn.Parameter(
                torch.randn(1, channels, spatial_size, spatial_size) * 0.02
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Add positional embeddings to input.

        Args:
            x:
                Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor:
                Tensor with positional embeddings added, shape
                `(B, C, H, W)`.
        '''
        _, _, h, w = x.shape
        assert isinstance(self.pos_embed, torch.nn.Parameter),\
            "pos_embed must be a learnable parameter"
        pos = self.pos_embed

        if (h, w) != (self.pos_embed.shape[2], self.pos_embed.shape[3]):
            pos = torch.nn.functional.interpolate(
                pos,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

        return x + pos

class _TransformerBlock(torch.nn.Module):
    '''
    Single transformer block with multi-head self-attention and MLP.

    Implements a pre-norm Vision Transformer block with residual
    multi-head self-attention and feed-forward layers.
    '''

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ):
        '''
        Initialize transformer block layers.

        Args:
            channels:
                Number of input and output channels.
            num_heads:
                Number of attention heads.
            mlp_ratio:
                Expansion ratio for hidden dimension in the MLP.
            dropout:
                Dropout probability applied in the MLP.
            attn_dropout:
                Attention dropout probability.
        '''
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm1 = torch.nn.LayerNorm(channels)
        self.attn = torch.nn.MultiheadAttention(
            channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )

        self.norm2 = torch.nn.LayerNorm(channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels, mlp_hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_hidden, channels),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply transformer block.

        Args:
            x:
                Input tensor of shape `(B, C, H, W)`.

        Returns:
            torch.Tensor:
                Output tensor of shape `(B, C, H, W)`.
        '''
        b, c, h, w = x.shape

        x_flat = x.flatten(2).transpose(1, 2)

        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        x_norm = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm)
        x_flat = x_flat + mlp_out

        x = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return x
