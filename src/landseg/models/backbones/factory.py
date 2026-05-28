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

'''Factory for backbone architectures.'''

# local imports
import landseg.models.backbones as backbones
import landseg.models.backbones.unet as unet
import landseg.models.backbones.unet.components as components

def build_backbone(
    in_ch: int,
    input_size: int,
    backbone_config: unet.BackboneConfig,
    bottleneck_config: components.BottleneckConfig
) -> backbones.Backbone:
    '''doc'''

    # core UNet body without bottleneck
    match backbone_config.body:
        case 'unet': backbone = backbones.UNet(in_ch, backbone_config)
        case 'unetpp': backbone = backbones.UNetPP(in_ch, backbone_config)
        case 'unetppp': backbone = backbones.UNetPPP(in_ch, backbone_config)
        case _: raise ValueError(f'Invalid backbone body: {backbone_config.body}')

    # bottleneck specs from the backbone
    if input_size % backbone.spatial_divisor != 0:
        raise ValueError(
            f'Input size {input_size} is not divisible by'
            f'backbone spatial divisor {backbone.spatial_divisor}.'
        )
    spatial_size = input_size // backbone.spatial_divisor

    # build bottleneck module
    backbone.build_bottleneck(bottleneck_config, spatial_size)

    # return
    return backbone
