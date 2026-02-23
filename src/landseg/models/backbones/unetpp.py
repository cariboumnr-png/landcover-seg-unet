'''
UNet++ backbone with nested skip connections.

**Overview**
UNet++ augments the standard UNet with densely connected nested skip paths
to reduce the semantic gap between encoder and decoder representations.
At each encoder depth `i` and refinement stage `j`, a node `X_{i,j}` is
computed by combining:
- the upsampled feature from the layer below (`X_{i+1, j-1}`),
- the original encoder skip (`X_{i,0}`),
- and all previous refinement nodes at the same depth (`X_{i,1}`, ...).

**Design points**
- 4-level encoder (identical to UNet) producing X_{i,0} features.
- Nested decoder computes X_{i,1}, X_{i,2}, X_{i,3}, and final X_{0,4}.
- All decoder nodes emit `base_ch` channels for consistent concatenation.
- Drop-in compatible with heads expecting `base_ch` width.

**Expected tensor shapes**
- Input:  (N, C_in, H, W)
- Encoder levels: spatial size halves per level (H/2ᵏ, W/2ᵏ), channels
  typically double per level.
- Decoder levels: spatial size increases via bilinear upsampling;
  concatenations occur along the channel dimension.
- Output: (N, base_ch, H, W) as the decoded backbone feature.

**Notes**
- This does not include deep supervision heads; it outputs X_{0,4}.
- Convolution weights are Kaiming-initialized for ReLU activations.
'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.backbones as backbones

class UNetPP(backbones.Backbone):
    '''UNet++ backbone with nested dense skip refinements.

    This class constructs a 4-level encoder identical to UNet, followed by
    a nested decoder where each layer computes intermediate refinement
    nodes (`X_{i,j}`) before producing the final decoded features.

    **Components**
    - inc: initial DoubleConv to lift input C_in → base_ch.
    - downs: four Downsample blocks (x2 channels, ×½ spatial size each).
    - nodes: ModuleDict of DoubleConv nodes that implement UNet++'s
      `X_{i,j}` refinements at multiple depths/stages.
    - ups: bilinear upsamplers used to align feature resolutions.

    **Notes**
    - The backbone's `out_channels` equals `base_ch`.
    - Skip concatenations are along channel dimension.
    - Convolution weights are Kaiming initialized (ReLU-friendly).
    '''

    # module aliases
    DC = backbones.DoubleConv
    DS = backbones.Downsample
    US = torch.nn.Upsample

    def __init__(self, in_ch: int, base_ch: int, **kwargs):
        '''
        Construct the UNet++ nested-skip backbone.

        **Architecture**

        Encoder (X_{i,0}):
            inc  → X_{0,0}
            down → X_{1,0}
            down → X_{2,0}
            down → X_{3,0}
            down → X_{4,0}

        Nested decoder:
            j = 1: X_{3,1}, X_{2,1}, X_{1,1}, X_{0,1}
            j = 2: X_{2,2}, X_{1,2}, X_{0,2}
            j = 3: X_{1,3}, X_{0,3}
            j = 4: X_{0,4} (final output)

        Args:
            in_ch: Number of input channels.
            base_ch: Base number of feature channels. Deeper layers use
                multiples of this.
            **kwargs: Additional options pass to convolutional blocks.
                see `models.backbones.DoubleConv`.

        **Notes**
        - All decoder nodes output `base_ch` channels which simplifies
          concatenation logic.
        - This version does not include deep-supervision heads; it
          produces only X_{0,4}.
        '''

        super().__init__()
        self._out_channels = base_ch # conforming to base class
        ch = base_ch # alias base_ch -> ch

        # initial convolution block with no norm
        self.inc = self.DC(in_ch, ch, norm=None, **kwargs)
        # 4 downsample encoders
        self.downs = torch.nn.ModuleList([
            self.DS(ch,     ch * 2,  **kwargs),
            self.DS(ch * 2, ch * 4,  **kwargs),
            self.DS(ch * 4, ch * 8,  **kwargs),
            self.DS(ch * 8, ch * 16, **kwargs)
        ])

        # decoders (nested refinement) - every nested node becomes ch channels
        # channel size helper dict
        chs = {0: ch, 1: ch * 2, 2: ch * 4, 3: ch * 8, 4: ch * 16}
        self.nodes = torch.nn.ModuleDict({
        # Level 1 nested (j=1)
            'x01': self.DC(chs[0]     + chs[1], chs[0], **kwargs),
            'x11': self.DC(chs[1]     + chs[2], chs[1], **kwargs),
            'x21': self.DC(chs[2]     + chs[3], chs[2], **kwargs),
            'x31': self.DC(chs[3]     + chs[4], chs[3], **kwargs),
            # Level 2 nested (j=2)
            'x02': self.DC(chs[0] * 2 + chs[1], chs[0], **kwargs),
            'x12': self.DC(chs[1] * 2 + chs[2], chs[1], **kwargs),
            'x22': self.DC(chs[2] * 2 + chs[3], chs[2], **kwargs),
            # Level 3 nested (j=3)
            'x03': self.DC(chs[0] * 3 + chs[1], chs[0], **kwargs),
            'x13': self.DC(chs[1] * 3 + chs[2], chs[1], **kwargs),
            # Level 4 nested (final output j=4)
            'x04': self.DC(chs[0] * 4 + chs[1], chs[0], **kwargs)
        })

        # upsamplers for backbone resolution
        self.ups = torch.nn.ModuleList([
            self.US(scale_factor=2, mode='bilinear', align_corners=False),
            self.US(scale_factor=2, mode='bilinear', align_corners=False),
            self.US(scale_factor=2, mode='bilinear', align_corners=False),
            self.US(scale_factor=2, mode='bilinear', align_corners=False)
        ])

        # Kaiming weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''Return 5-level encoder features.'''

        x0_0 = self.inc(x)
        x1_0 = self.downs[0](x0_0)
        x2_0 = self.downs[1](x1_0)
        x3_0 = self.downs[2](x2_0)
        x4_0 = self.downs[3](x3_0)

        return x0_0, x1_0, x2_0, x3_0, x4_0

    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        '''Decode nested UNet++ refinement graph.'''

        # unpack
        x = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}}
        x[0][0], x[1][0], x[2][0], x[3][0], x[4][0] = xs

        # aliases
        cat = self._cat
        u = self.ups

        # ---------------- j = 1 ----------------
        x[3][1] = cat('x31', x[3][0], u[3](x[4][0]))
        x[2][1] = cat('x21', x[2][0], u[2](x[3][0]))
        x[1][1] = cat('x11', x[1][0], u[1](x[2][0]))
        x[0][1] = cat('x01', x[0][0], u[0](x[1][0]))

        # ---------------- j = 2 ----------------
        x[2][2] = cat('x22', x[2][0], x[2][1], u[2](x[3][1]))
        x[1][2] = cat('x12', x[1][0], x[1][1], u[1](x[2][1]))
        x[0][2] = cat('x02', x[0][0], x[0][1], u[0](x[1][1]))

        # ---------------- j = 3 ----------------
        x[1][3] = cat('x13', x[1][0], x[1][1], x[1][2], u[1](x[2][2]))
        x[0][3] = cat('x03', x[0][0], x[0][1], x[0][2], u[0](x[1][2]))

        # ------------ j = 4 (final) ------------
        x[0][4] = cat('x04', x[0][0], x[0][1], x[0][2], x[0][3], u[0](x[1][3]))

        # return
        return x[0][4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''End-to-end UNet++ forward pass.'''

        xs = self.encode(x)
        return self.decode(xs)

    def _cat(self, node: str, *tensors: torch.Tensor) -> torch.Tensor:
        '''Concatenate then apply the named DoubleConv node.'''

        x = torch.cat(tensors, dim=1)
        return self.nodes[node](x)

    @property
    def out_channels(self) -> int:
        '''Return channel width of final decoded representation.'''
        return self._out_channels
