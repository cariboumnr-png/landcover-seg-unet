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
Top-level namespace for `landseg.models.backbones.unet.components`.

Exposes UNet convolutional blocks, encoder stages, bottleneck modules,
and configuration protocols via lazy resolution to keep import order
simple and circular-free.

Public APIs:
    - `BaseBottleneck`: Abstract base class for UNet bottlenecks.
    - `BottleneckConfig`: Protocol for bottleneck configuration.
    - `ConvolutionParameters`: Protocol for convolution block options.
    - `DoubleConv`: Stacked two-stage convolution-norm-ReLU block.
    - `Downsample`: Max-pool downsampling stage with `DoubleConv`.
    - `HybridBottleneck`: Combined convolutional-transformer bottleneck.
    - `TransformerBottleneck`: Vision Transformer bottleneck stack.
    - `TransformerParameters`: Protocol for transformer block options.
    - `UNetBottleneck`: Standard convolutional UNet bottleneck block.
    - `UNetEncoders`: Multi-stage UNet encoder pathway.
    - `Upsample`: Bilinear upsampling and skip-fusion stage.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'BaseBottleneck',
    'BottleneckConfig',
    'ConvolutionParameters',
    'DoubleConv',
    'Downsample',
    'HybridBottleneck',
    'TransformerBottleneck',
    'TransformerParameters',
    'UNetBottleneck',
    'UNetEncoders',
    'Upsample',
]


# for static check
if typing.TYPE_CHECKING:
    from .bottlenecks import (
        BaseBottleneck,
        HybridBottleneck,
        TransformerBottleneck,
        UNetBottleneck,
    )
    from .configs import (
        BottleneckConfig,
        ConvolutionParameters,
        TransformerParameters,
    )
    from .conv_blocks import (
        DoubleConv,
        Downsample,
        Upsample,
    )
    from .encoders import UNetEncoders


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.bottlenecks': [
        'BaseBottleneck',
        'HybridBottleneck',
        'TransformerBottleneck',
        'UNetBottleneck',
    ],
    '.configs': [
        'BottleneckConfig',
        'ConvolutionParameters',
        'TransformerParameters',
    ],
    '.conv_blocks': [
        'DoubleConv',
        'Downsample',
        'Upsample',
    ],
    '.encoders': ['UNetEncoders'],
})
