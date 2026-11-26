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
UNet++ backbone with nested skip connections.

Overview:
    UNet++ augments the standard UNet with densely connected nested skip
    paths to reduce the semantic gap between encoder and decoder
    representations. At each encoder depth `i` and refinement stage `j`,
    a node `X_{i,j}` is computed by combining:
    - the upsampled feature from the layer below (`X_{i+1, j-1}`),
    - the original encoder skip (`X_{i,0}`),
    - previous refinement nodes at the same depth (`X_{i,1}`, ...).

Design Points:
    - 4-level encoder matching `UNet` producing `X_{i,0}` features.
    - Nested decoder computes `X_{i,1}`, `X_{i,2}`, `X_{i,3}`, and final
      `X_{0,4}`.
    - Full-resolution nodes emit `base_ch` channels for head input.
    - Drop-in compatible with heads expecting `base_ch` width.

Expected Tensor Shapes:
    - Input: `(N, C_in, H, W)`
    - Encoder levels: spatial size halves per level (`H/2ᵏ`, `W/2ᵏ`),
      channels double per level.
    - Decoder levels: spatial size increases via bilinear upsampling;
      concatenations occur along the channel dimension.
    - Output: `(N, base_ch, H, W)` as the decoded backbone feature.

Notes:
    - Does not include deep supervision heads; outputs `X_{0,4}` only.
    - Convolution weights are Kaiming-initialized for ReLU activations.
'''

# third-party imports
import torch
import torch.nn
# local imports
import landseg.models.backbones.unet.body as body
import landseg.models.backbones.unet.components as components

class UNetPP(body.UNetBackbone):
    '''
    UNet++ backbone with nested dense skip refinements.

    Constructs a 4-level encoder identical to `UNet`, followed by a
    nested decoder where each layer computes intermediate refinement
    nodes (`X_{i,j}`) before producing the final decoded features.

    Components:
        - `inc`: Initial `DoubleConv` lifting `C_in` to `base_ch`.
        - `downs`: Four `Downsample` stages (`2x` ch, `0.5x` size).
        - `nodes`: `ModuleDict` of `DoubleConv` refinement nodes
          `X_{i,j}` across decoder depths and stages.
        - `ups`: Bilinear upsamplers aligning feature resolutions.

    Notes:
        - The backbone's `out_channels` equals `base_ch`.
        - Skip concatenations are along the channel dimension.
        - Convolution weights are Kaiming-initialized for ReLU.
    '''

    def __init__(
        self,
        in_ch: int,
        config: body.UNetBodyConfig
    ):
        '''
        Construct the UNet++ nested-skip backbone.

        Initializes the 4-level encoder (`X_{i,0}`) and builds the
        dense nested decoder graph (`X_{i,j}` for `j = 1..4`) with
        bilinear upsamplers and Kaiming normal `Conv2d` weights.

        Args:
            in_ch:
                Number of input channels.
            config:
                `UNetBodyConfig` specifying base channel width,
                encoder parameters, and `nodes_conv_params`.
        '''
        super().__init__(in_ch, config)
        self._out_channels = ch = config.base_ch

        # decoders (nested refinement) - every nested node becomes ch channels
        # channel size helper dict
        chs = {0: ch, 1: ch * 2, 2: ch * 4, 3: ch * 8, 4: ch * 16}
        assert config.nodes_conv_params
        DC = components.DoubleConv
        self.nodes = torch.nn.ModuleDict({
        # Level 1 nested (j=1)
            'x01': DC(chs[0]     + chs[1], chs[0], config.nodes_conv_params),
            'x11': DC(chs[1]     + chs[2], chs[1], config.nodes_conv_params),
            'x21': DC(chs[2]     + chs[3], chs[2], config.nodes_conv_params),
            'x31': DC(chs[3]     + chs[4], chs[3], config.nodes_conv_params),
            # Level 2 nested (j=2)
            'x02': DC(chs[0] * 2 + chs[1], chs[0], config.nodes_conv_params),
            'x12': DC(chs[1] * 2 + chs[2], chs[1], config.nodes_conv_params),
            'x22': DC(chs[2] * 2 + chs[3], chs[2], config.nodes_conv_params),
            # Level 3 nested (j=3)
            'x03': DC(chs[0] * 3 + chs[1], chs[0], config.nodes_conv_params),
            'x13': DC(chs[1] * 3 + chs[2], chs[1], config.nodes_conv_params),
            # Level 4 nested (final output j=4)
            'x04': DC(chs[0] * 4 + chs[1], chs[0], config.nodes_conv_params)
        })

        # upsamplers for backbone resolution
        US = torch.nn.Upsample
        self.ups = torch.nn.ModuleList([
            US(scale_factor=2, mode='bilinear', align_corners=False),
            US(scale_factor=2, mode='bilinear', align_corners=False),
            US(scale_factor=2, mode='bilinear', align_corners=False),
            US(scale_factor=2, mode='bilinear', align_corners=False)
        ])

        # Kaiming weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    @property
    def bottleneck_ch(self) -> int:
        '''Return the bottleneck channel number.'''
        return self._out_channels * 16

    @property
    def out_channels(self) -> int:
        '''Return channel width of final decoded representation.'''
        return self._out_channels

    @property
    def spatial_divisor(self) -> int:
        '''Return minimum spatial divisor induced by the encoder.'''
        return 2 ** len(self.downs)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        '''
        Run the encoder path and bottleneck to extract 5-level features.

        Args:
            x:
                Input tensor of shape `(B, C_in, H, W)`.

        Returns:
            tuple[torch.Tensor, ...]:
                Five multi-scale feature tensors `(x1, x2, x3, x4, xb)`
                ordered from finest to bottleneck resolution.
        '''
        # unpack the 5 levels
        x1, x2, x3, x4, x5 = self.downs(x)
        # apply bottleneck to the deepest encoder feature
        xb = self.bottleneck(x5)
        return x1, x2, x3, x4, xb

    def decode(self, xs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        '''
        Decode encoder features through the nested UNet++ skip graph.

        Args:
            xs:
                Five-element tuple `(x1, x2, x3, x4, xb)` produced by
                `encode`.

        Returns:
            torch.Tensor:
                Final refined feature map `X_{0,4}` of shape
                `(B, out_channels, H, W)`.
        '''
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
        '''
        Compute the UNet++ feature map at input spatial resolution.

        Args:
            x:
                Input tensor of shape `(B, C_in, H, W)`.

        Returns:
            torch.Tensor:
                Output feature map of shape `(B, out_channels, H, W)`.
        '''
        xs = self.encode(x)
        return self.decode(xs)

    def _cat(self, node: str, *tensors: torch.Tensor) -> torch.Tensor:
        '''Concatenate tensors and apply the named `DoubleConv` node.'''
        x = torch.cat(tensors, dim=1)
        return self.nodes[node](x)
