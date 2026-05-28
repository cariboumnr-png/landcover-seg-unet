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

# pylint: disable=missing-function-docstring

'''Backbone components configurations.'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg.models.backbones.unet.components as components

class BackboneConfig(typing.Protocol):
    '''Typed container for backbone convolution configuration.'''
    @property
    def body(self) -> str: ...
    @property
    def base_ch(self) -> int: ...
    @property
    def encoder_conv_params(self) -> components.ConvolutionParameters: ...
    @property
    def nodes_conv_params(self) -> components.ConvolutionParameters | None: ...
    @property
    def decoder_conv_params(self) -> components.ConvolutionParameters | None: ...
