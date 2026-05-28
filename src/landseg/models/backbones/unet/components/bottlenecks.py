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
Transformer bottleneck module for UNet.

This module provides a drop-in replacement for the standard convolutional
bottleneck in U-Net architectures. It uses Vision Transformer building
blocks (multi-head self-attention, feed-forward networks) to capture
long-range dependencies at the bottleneck, which can improve segmentation
performance especially on large-scale spatial patterns.

**Key Features**
- Vision Transformer-based bottleneck preserving spatial structure
- Learnable absolute positional embeddings for spatial context
- Multi-head self-attention with residual connections
- Feed-forward networks with gating (optional)
- Layer normalization for stable training
- Drop-in compatible with existing UNet implementations

**Design Points**
- Maintains input/output channel compatibility (in_ch == out_ch)
- Preserves spatial dimensions (no downsampling in bottleneck)
- Uses residual connections for gradient flow
- Supports variable patch-based processing or global attention
- Configurable number of transformer blocks and attention heads
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

    Defines the interface and common functionality for bottleneck modules
    used in UNet architectures. Subclasses must implement the forward
    method to process the input feature map and produce an output of
    identical shape.

    This class is not intended to be instantiated directly. It serves as
    a template for specific bottleneck implementations (e.g., pure
    transformer, hybrid conv-transformer).
    '''

    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Process input feature map through the bottleneck.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)

        Notes:
        - Subclasses must implement this method to define the bottleneck
          processing logic.
        - The output must have the same shape as the input to maintain
          compatibility with the UNet architecture.
        '''
        raise NotImplementedError

class UNetBottleneck(BaseBottleneck):
    '''
    Standard convolutional bottleneck for UNet.

    Implements the traditional double convolutional block used at the
    bottleneck of U-Net architectures. This serves as a baseline and can
    be replaced with more complex modules (e.g., transformer-based) if
    desired.

    Architecture:
    - DoubleConv with in_ch == out_ch, no downsampling.
    - Optional normalization and dropout based on provided kwargs.
    '''

    def __init__(
        self,
        in_ch: int,
        params: components.ConvolutionParameters
    ):
        super().__init__(in_ch)
        self.double_conv = components.DoubleConv(in_ch, in_ch, params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class TransformerBottleneck(BaseBottleneck):
    '''
    Transformer-based bottleneck for U-Net.

    Replaces the standard convolutional bottleneck with a stack of
    transformer blocks, enabling long-range dependency modeling while
    preserving spatial structure.

    This module is designed as a drop-in replacement for the bottleneck
    layer in UNet architectures. It accepts a single input tensor and
    produces output of identical shape.

    Args:
        in_channels: Number of input channels.
        num_blocks: Number of transformer blocks to stack.
        num_heads: Number of attention heads per block.
        mlp_ratio: Feed-forward network expansion ratio.
        dropout: Dropout probability.
        attn_dropout: Attention-specific dropout probability.

    Example:
        Replace a UNet bottleneck:

        # Original
        bottleneck = DoubleConv(ch*16, ch*16)

        # With transformer
        bottleneck = TransformerBottleneck(ch*16, num_blocks=4)

        # Usage remains identical
        x_bottleneck = bottleneck(x)
    '''

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        num_blocks: int,
        params: components.TransformerParameters
    ):
        super().__init__(in_channels)
        self.in_ch = in_channels

        self.pos_embed = _PositionalEmbedding(in_channels, spatial_size)

        self.blocks = torch.nn.ModuleList([
            _TransformerBlock(
                in_channels,
                num_heads=params.num_heads,
                mlp_ratio=params.mlp_ratio,
                dropout=params.dropout,
                attn_dropout=params.attn_dropout,
            )
            for _ in range(num_blocks)
        ])

        self.norm_out = torch.nn.LayerNorm(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply transformer bottleneck.

        Args:
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
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
    Hybrid bottleneck combining convolution and transformer.

    Provides a middle ground between purely convolutional and purely
    transformer-based bottlenecks. Useful for computational efficiency
    and can sometimes provide better inductive bias than pure transformers.

    Args:
        in_channels: Number of input/output channels.
        num_conv_blocks: Number of convolutional blocks at start.
        num_transformer_blocks: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Feed-forward expansion ratio.
        dropout: Dropout probability.
        attn_dropout: Attention-specific dropout probability.
        **conv_params: Additional arguments passed to convolutional blocks.

    Example:
        # Use hybrid for efficiency
        bottleneck = HybridBottleneck(
            ch*16,
            num_conv_blocks=1,
            num_transformer_blocks=2,
            num_heads=4
        )
    '''

    def __init__(
        self,
        in_channels: int,
        spatial_size: int,
        config: components.HybridBottleneckConfig,

    ):
        super().__init__(in_channels)
        self.in_ch = in_channels

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
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
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
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor with positional embeddings added, shape (B, C, H, W)
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

    Implements a standard Vision Transformer block with:
    - Layer normalization before attention and MLP (pre-norm)
    - Multi-head self-attention with residual connection
    - Position-wise feed-forward network with residual connection
    - Optional dropout for regularization

    Args:
        channels: Number of input/output channels.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for hidden dimension in MLP
            (hidden_dim = channels * mlp_ratio).
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
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
            x: Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
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
