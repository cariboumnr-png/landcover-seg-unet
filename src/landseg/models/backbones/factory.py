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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Factory for backbone architectures.

This module provides a single factory function for instantiating
UNet-based backbone implementations. The factory validates input
spatial constraints and composes encoder, bottleneck, and decoder
components into a complete feature extractor.
'''

# standard imports
import typing
# local imports
import landseg.models.backbones.unet.body as body
import landseg.models.backbones.unet.components as components

class UNetBackboneConfig(typing.Protocol):
    @property
    def body(self) -> body.UNetBodyConfig: ...
    @property
    def bottleneck(self) -> components.BottleneckConfig: ...

def build_unet_backbone(
    in_ch: int,
    input_size: int,
    config: UNetBackboneConfig
) -> body.UNetBackbone:
    '''
    Instantiate a UNet backbone with configured body and bottleneck.

    This factory constructs a complete UNet-based feature extractor by:
    1. Selecting a backbone body variant (UNet, UNet++, or UNet+++).
    2. Validating input spatial size against backbone divisibility.
    3. Instantiating and configuring the bottleneck module.

    Args:
        in_ch: Number of input channels.
        input_size: Spatial dimension (assumes square inputs).
        config: UNetBackboneConfig specifying body and bottleneck
            configuration.

    Returns:
        Fully constructed UNetBackbone instance ready for encoding
        and decoding operations.

    Raises:
        ValueError: If input size is not divisible by backbone's
            spatial divisor.
    '''

    # aliases
    unetbody_cfg = config.body
    bottleneck_cfg = config.bottleneck

    # core UNet body without bottleneck
    match unetbody_cfg.body:
        case 'unet': backbone = body.UNet(in_ch, unetbody_cfg)
        case 'unetpp': backbone = body.UNetPP(in_ch, unetbody_cfg)
        case 'unetppp': backbone = body.UNetPPP(in_ch, unetbody_cfg)
        case _: raise ValueError(f'Invalid backbone body: {unetbody_cfg.body}')

    # bottleneck specs from the backbone
    if input_size % backbone.spatial_divisor != 0:
        raise ValueError(
            f'Input size {input_size} is not divisible by'
            f'backbone spatial divisor {backbone.spatial_divisor}.'
        )
    spatial_size = input_size // backbone.spatial_divisor

    # build bottleneck module
    backbone.build_bottleneck(bottleneck_cfg, spatial_size)

    # return
    return backbone
