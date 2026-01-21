'''
Standard UNet architecture.

A compact yet extensible implementation of the U-Net architecture for
dense, per-pixel prediction tasks (e.g., semantic segmentation, binary
masks, and contour detection).

**Overview**
U-Net is a fully convolutional encoder-decoder network with long skip
connections. The encoder contracts spatial dimensions to extract
hierarchical, high-level features; the decoder expands them while
recovering fine details using skip connections from matching encoder
stages. This topology enables accurate localization with modest data
requirements.

**Key design points in this module**
- Encoder (contracting path): stacks of convolutional blocks followed by
  downsampling to progressively increase receptive field.
- Bottleneck: deepest representation capturing global context.
- Decoder (expanding path): upsampling blocks that fuse encoder features
  via channel-wise concatenation (skip connections).
- Output head(s): 1x1 convolutions to produce per-pixel logits or
  intermediate feature maps.
- Normalization: configurable per-block (none, batch, group-as-layer).
- Initialization: Kaiming initialization for Conv2d weights (ReLU).

**Expected tensor shapes**
- Input:  `(N, C_in, H, W)`
- Encoder levels: spatial size halves per downsample (H/2ᵏ, W/2ᵏ),
  channels typically double.
- Decoder levels: spatial size doubles per upsample, channels reduce,
  with skip concatenations on channel dimension.
- Output: `(N, C_out, H, W)` when a head is attached externally.

**Extension points**
- Swap normalization via keyword arguments.
- Replace upsampling with transposed convolutions if learned upsampling
  is desired.
- Add heads (e.g., segmentation, edge, uncertainty) atop the UNet body.

This module exposes a `UNet` backbone and a small family of internal
building blocks intended for composition inside more complex models.
'''

# third-part imports
import torch
import torch.nn
# local imports
import models.backbones

class UNet(models.backbones.Backbone):
    '''
    UNet backbone implementing an encoder-decoder with skip connections.

    This class provides the core U-Net body—initial conv, multi-stage
    downsampling path, bottleneck, and symmetric upsampling path that
    concatenates encoder features at matching resolutions. Agnostic
    to the final prediction head, it can be reused across tasks (e.g.,
    binary/semantic segmentation, regression heatmaps) by attaching an
    appropriate 1x1 convolution head externally.

    **Components**
    - `inc`: init double-conv block to lift input channels to `base_ch`.
    - `downs`: Four `DownsampleBlock` modules (x2 ch | ÷2 size).
    - `bottleneck`: a DoubleConvBlock` operating at the coarsest scale.
    - `ups`: Four `UpsampleBlock` modules (feature fusion + upsample).

    **Notes**
    - Skip connections are concatenated along the channel dimension.
    - Convolution weights are Kaiming-initialized (fan-out, ReLU).
    - The backbone's `out_channels` = `base_ch` (decoder's final width).

    See `__init__` for construction details and configuration options.
    '''

    # core UNet body
    def __init__(
            self,
            in_ch: int,
            base_ch: int,
            **kwargs
        ):
        '''
        Construct UNet body with configurable normalization and dropout.

        **Architecture**:
        - Initial block: Converts input channels to `base_ch`.
        - Down blocks: Halve spatial resolution and double channels at
        each step.
        - Bottom block: Deepest representation with `16 x base_ch`
        channels.
        - Up blocks: Double spatial resolution and reduce channels,
        concatenating skips.
        - Output heads: Apply 1x1 convolutions to produce per-pixel
        class logits.

        Args:
            in_ch: Number of input channels.
            base_ch: Base number of feature channels. Deeper layers use
                multiples of this.
            **kwargs: Additional options pass to convolutional blocks.
                - `norm` (str | None): normalization kind; one of
                    {'bn', 'gn', 'ln', None}. Default: 'gn'.
                - `gn_groups` (int): number of groups when `norm='gn'`.
                    Automatically reduced to a divisor of channels.
                    Default: 8.
                - `p_drop` (float): dropout probability used in
                    `DoubleConvBlock`. Default: 0.05.

        **Notes**:
        - Weight initialization uses Kaiming normal suited for ReLU.
        - `out_channels` property exposes the decoder's final width
            (`base_ch`).
        '''

        super().__init__()
         # conforming to base class
        self._out_channels = base_ch

        # convolution layers (alias base_ch -> ch)
        ch = base_ch
        # initial convolution block with no norm
        self.inc = models.backbones.DoubleConv(in_ch, ch, norm=None, **kwargs)
        # 4 downs
        self.downs = torch.nn.ModuleList([
            models.backbones.Downsample(ch,   ch*2,  **kwargs),
            models.backbones.Downsample(ch*2, ch*4,  **kwargs),
            models.backbones.Downsample(ch*4, ch*8,  **kwargs),
            models.backbones.Downsample(ch*8, ch*16, **kwargs),
        ])
        # bottleneck
        self.bottleneck = models.backbones.DoubleConv(ch*16, ch*16, **kwargs)
        # 4 ups
        self.ups = torch.nn.ModuleList([
            models.backbones.Upsample(ch*16 + ch*8, ch*8, **kwargs),
            models.backbones.Upsample(ch*8  + ch*4, ch*4, **kwargs),
            models.backbones.Upsample(ch*4  + ch*2, ch*2, **kwargs),
            models.backbones.Upsample(ch*2  + ch,   ch,   **kwargs)
        ])

        # Kaiming weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''Run the contracting path and return encoder features.'''

        x1 = self.inc(x)                # H     in_ch ->  b
        x2 = self.downs[0](x1)          # H/2   b     ->  2b
        x3 = self.downs[1](x2)          # H/4   2b    ->  4b
        x4 = self.downs[2](x3)          # H/8   4b    ->  8b
        x5 = self.downs[3](x4)          # H/16  8b    ->  16b
        xb = self.bottleneck(x5)        # H/16  16b   --  16b
        return x1, x2, x3, x4, xb

    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        '''Fuse skips and upsample through the expanding path.'''

        x1, x2, x3, x4, xb = xs
        x = self.ups[0](xb, x4)
        x = self.ups[1](x, x3)
        x = self.ups[2](x, x2)
        x = self.ups[3](x, x1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Compute the UNet feature map at input resolution.'''

        x1, x2, x3, x4, xb = self.encode(x)
        x = self.decode((x1, x2, x3, x4, xb))
        return x

    @property
    def out_channels(self) -> int:
        '''Return the channel width of the backbone output.'''
        return self._out_channels
