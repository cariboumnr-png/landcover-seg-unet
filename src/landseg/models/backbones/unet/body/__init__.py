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
Top-level namespace for `landseg.models.backbones.unet.body`.

Exposes UNet body base classes, configuration contracts, and concrete
architectures via lazy resolution to keep import order simple and
circular-free.

Public APIs:
    - `UNet`: Standard UNet backbone with skip connections.
    - `UNetBackbone`: Abstract base class for UNet backbones.
    - `UNetBodyConfig`: Protocol for UNet body configuration.
    - `UNetPP`: UNet++ backbone with nested dense skip connections.
    - `UNetPPP`: UNet3+ backbone with full-scale skip aggregation.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'UNet',
    'UNetBackbone',
    'UNetBodyConfig',
    'UNetPP',
    'UNetPPP',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import UNetBackbone
    from .configs import UNetBodyConfig
    from .unet import UNet
    from .unetpp import UNetPP
    from .unetppp import UNetPPP


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.base': ['UNetBackbone'],
    '.configs': ['UNetBodyConfig'],
    '.unet': ['UNet'],
    '.unetpp': ['UNetPP'],
    '.unetppp': ['UNetPPP'],
})
