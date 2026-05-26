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

# pylint: disable=not-callable
# Note: this is to suppress pylint complaining `avg_pool2d`` not callable,
# which **IS** callable

'''
UNet3+ backbone implementing full-scale dense skip aggregation.

This module defines a UNet3+ encoder-decoder architecture with
full-resolution feature fusion across all encoder stages. Unlike
standard U-Net and UNet++, UNet3+ removes strict scale-by-scale
decoding and instead aggregates projected features from all encoder
levels at each decoder stage using explicit spatial alignment.

**Core Design**
- Encoder: 4-stage hierarchical downsampling (x2 per stage)
- Feature projection: all encoder outputs mapped to a unified width
- Decoder: full-scale aggregation of all encoder + intermediate decoder
  features at each stage
- Resampling: explicit interpolation/pooling for cross-scale alignment

**Key Properties**
- Full-scale skip connections across all resolutions
- Unified channel space for all encoder features
- Direction-aware resizing (upsampling + exact downsampling support)
- Decoder operates on dynamically aligned feature grids rather than a
  fixed spatial pyramid

**Differences from UNet / UNet++**
- UNet: single-path symmetric encoder-decoder
- UNet++: nested, progressively refined skip nodes
- UNet3+: full-resolution feature fusion with global skip aggregation

**Notes**
- Spatial hierarchy is defined by the encoder only (x2 per level)
- Decoder does not impose additional strict divisibility constraints
- Output channel width equals `base_ch`
'''

# third-party imports
import torch
import torch.nn as nn
import torch.nn.functional
# local imports
import landseg.models.backbones.unet as unet
import landseg.models.backbones.unet.components as components

class UNetPPP(unet.UNetBackbone):
    '''
    Canonical UNet3+ backbone.

    Characteristics:
    - full-scale skip aggregation across encoder levels
    - unified projection width for all encoder features
    - dense multi-resolution decoder connectivity
    - explicit resizing between all feature scales (upsampling + optional
        pooled downsampling)
    - encoder defines base spatial hierarchy; decoder operates on
        interpolated feature grids

    Notes:
    - Output channel width equals `base_ch`
    - Spatial alignment is handled via runtime interpolation, not fixed
        tensor shapes
    - Encoder depth defines the minimum spatial divisor constraint
    '''

    # aliases
    DC = components.DoubleConv
    DS = components.Downsample

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        **kwargs,
    ) -> None:
        '''
        Initialize backbone.
        '''

        super().__init__()
        # unified aggregation width
        agg_ch = ch = base_ch
        self._out_channels = base_ch

        # ------------------------------------------------------------------ #
        # encoder
        # ------------------------------------------------------------------ #

        self.inc = self.DC(in_ch, ch, norm=None, p_drop=0.0)
        
        self.downs = nn.ModuleList([
            self.DS(ch,      ch * 2,  **kwargs.get('downs', {})),
            self.DS(ch * 2,  ch * 4,  **kwargs.get('downs', {})),
            self.DS(ch * 4,  ch * 8,  **kwargs.get('downs', {})),
            self.DS(ch * 8,  ch * 16, **kwargs.get('downs', {})),
        ])

        # ------------------------------------------------------------------ #
        # encoder projections
        # all features projected to common aggregation width
        # ------------------------------------------------------------------ #

        self.projs = nn.ModuleList([
            nn.Conv2d(ch,      agg_ch, 1),
            nn.Conv2d(ch * 2,  agg_ch, 1),
            nn.Conv2d(ch * 4,  agg_ch, 1),
            nn.Conv2d(ch * 8,  agg_ch, 1),
            nn.Conv2d(ch * 16, agg_ch, 1),
        ])

        # ------------------------------------------------------------------ #
        # decoder blocks
        # each stage aggregates 5 tensors
        # ------------------------------------------------------------------ #

        self.ups = nn.ModuleList([
            self.DC(agg_ch * 5, agg_ch, **kwargs.get('nodes', {})),
            self.DC(agg_ch * 5, agg_ch, **kwargs.get('nodes', {})),
            self.DC(agg_ch * 5, agg_ch, **kwargs.get('nodes', {})),
            self.DC(agg_ch * 5, agg_ch, **kwargs.get('nodes', {})),
        ])

        # Kaiming weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    @property
    def bottleneck_ch(self) -> int:
        '''Return the bottleneck channel number.'''
        return self._out_channels # same

    @property
    def out_channels(self) -> int:
        '''Number of output channels.'''
        return self._out_channels

    @property
    def spatial_divisor(self) -> int:
        '''Minimum spatial divisor induced by the encoder hierarchy.'''
        return 2 ** len(self.downs)

    def _resize(
        self,
        x: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        '''
        Direction-aware resize.

        - bilinear for upsampling
        - avg pool for exact integer downsampling
        - antialiased bilinear fallback otherwise
        '''

        h_in, w_in = x.shape[-2:]
        h_out, w_out = size

        # pure upsample
        if h_out >= h_in and w_out >= w_in:
            return torch.nn.functional.interpolate(
                x,
                size=(h_out, w_out),
                mode='bilinear',
                align_corners=False,
            )

        # exact integer downsample
        if h_in % h_out == 0 and w_in % w_out == 0:

            k_h = h_in // h_out
            k_w = w_in // w_out

            return torch.nn.functional.avg_pool2d(
                x,
                kernel_size=(k_h, k_w),
                stride=(k_h, k_w),
            )

        # fallback
        return torch.nn.functional.interpolate(
            x,
            size=(h_out, w_out),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        '''Run encoder path and return multi-scale feature hierarchy.'''

        x1 = self.inc(x)
        x2 = self.downs[0](x1)
        x3 = self.downs[1](x2)
        x4 = self.downs[2](x3)
        x5 = self.downs[3](x4)

        return x1, x2, x3, x4, x5

    def decode(
        self,
        xs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        '''UNet3+ full-scale feature aggregation decoder.'''

        x1, x2, x3, x4, x5 = xs

        # ------------------------------------------------------------------ #
        # projected encoder features
        # ------------------------------------------------------------------ #

        p1 = self.projs[0](x1)
        p2 = self.projs[1](x2)
        p3 = self.projs[2](x3)
        p4 = self.projs[3](x4)
        p5 = self.projs[4](x5)

        projs = [p1, p2, p3, p4, p5]

        sizes = [
            (x1.shape[-2], x1.shape[-1]),
            (x2.shape[-2], x2.shape[-1]),
            (x3.shape[-2], x3.shape[-1]),
            (x4.shape[-2], x4.shape[-1])
        ]

        # ------------------------------------------------------------------ #
        # resize cache
        # ------------------------------------------------------------------ #

        cache: dict[tuple[int, int], torch.Tensor] = {}

        def get(i: int, j: int) -> torch.Tensor:
            '''
            Resize projected feature i to decoder scale j.
            '''
            key = (i, j)

            if key not in cache:
                cache[key] = self._resize(
                    projs[i],
                    sizes[j],
                )

            return cache[key]

        # ------------------------------------------------------------------ #
        # d4 -> H/8
        # ------------------------------------------------------------------ #

        d4 = self.ups[0](torch.cat([
            get(0, 3),   # x1 -> H/8
            get(1, 3),   # x2 -> H/8
            get(2, 3),   # x3 -> H/8
            p4,           # x4 native
            get(4, 3),   # x5 -> H/8
        ], dim=1))

        # ------------------------------------------------------------------ #
        # d3 -> H/4
        # ------------------------------------------------------------------ #

        d3 = self.ups[1](torch.cat([
            get(0, 2),                     # x1
            get(1, 2),                     # x2
            p3,                            # x3
            self._resize(d4, sizes[2]),    # d4
            get(4, 2),                     # x5
        ], dim=1))

        # ------------------------------------------------------------------ #
        # d2 -> H/2
        # ------------------------------------------------------------------ #

        d2 = self.ups[2](torch.cat([
            get(0, 1),                     # x1
            p2,                            # x2
            self._resize(d3, sizes[1]),    # d3
            self._resize(d4, sizes[1]),    # d4
            get(4, 1),                     # x5
        ], dim=1))

        # ------------------------------------------------------------------ #
        # d1 -> H
        # ------------------------------------------------------------------ #

        d1 = self.ups[3](torch.cat([
            p1,                            # x1
            self._resize(d2, sizes[0]),    # d2
            self._resize(d3, sizes[0]),    # d3
            self._resize(d4, sizes[0]),    # d4
            get(4, 0),                     # x5
        ], dim=1))

        return d1

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        '''Compute UNet3+ feature map.'''

        xs = self.encode(x)
        return self.decode(xs)
