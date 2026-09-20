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

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BaseBottleneck',
    'DoubleConv',
    'Downsample',
    'HybridBottleneck',
    'TransformerBottleneck',
    'UNetBottleneck',
    'UNetEncoders',
    'Upsample',
    # typing
    'BottleneckConfig',
    'ConvolutionParameters',
    'TransformerParameters',
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
    from .encoders import (
        UNetEncoders,
    )


def __getattr__(name: str):
    if name in {
        'BaseBottleneck',
        'HybridBottleneck',
        'TransformerBottleneck',
        'UNetBottleneck',
    }:
        obj = importlib.import_module('.bottlenecks', __package__)
        return getattr(obj, name)

    if name in {
        'BottleneckConfig',
        'ConvolutionParameters',
        'TransformerParameters',
    }:
        obj = importlib.import_module('.configs', __package__)
        return getattr(obj, name)

    if name in {
        'DoubleConv',
        'Downsample',
        'Upsample',
    }:
        obj = importlib.import_module('.conv_blocks', __package__)
        return getattr(obj, name)

    if name in {'UNetEncoders'}:
        obj = importlib.import_module('.encoders', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
