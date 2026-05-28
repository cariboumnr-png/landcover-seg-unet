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

# standard imports
import typing
# local imports
import landseg.models.backbones as backbones
import landseg.models.backbones.unet.components as components

def build_backbone(
    in_ch: int,
    base_ch: int,
    backbone_body: typing.Literal['unet', 'unetpp', 'unetppp'],
    bottleneck_variant: typing.Literal['transformer', 'hybrid'] | None,
    conv_parameters: dict[str, typing.Any],
) -> backbones.Backbone:
    '''doc'''

    # bottleneck module
    match bottleneck_variant:
        case 'transformer':
            bottleneck = components.TransformerBottleneck(base_ch * 16)
        case 'hybrid':
            bottleneck = components.HybridBottleneck(base_ch * 16)
        case None:
            bottleneck = None
        case _:
            raise ValueError(f'Invalid bottleneck variant: {bottleneck_variant}')

    # core UNet body with bottleneck
    match backbone_body:
        case 'unet':
            backbone = backbones.UNet(in_ch, base_ch, bottleneck, **conv_parameters)
        case 'unetpp':
            backbone = backbones.UNetPP(in_ch, base_ch, bottleneck, **conv_parameters)
        case 'unetppp':
            backbone = backbones.UNetPPP(in_ch, base_ch, bottleneck, **conv_parameters)
        case _:
            raise ValueError(f'Invalid backbone body: {backbone_body}')

    # return
    return backbone
