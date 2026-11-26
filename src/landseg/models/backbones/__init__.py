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
Top-level namespace for `landseg.models.backbones`.

Exposes backbone interfaces, configuration contracts, and factory
builders via lazy resolution to keep import order simple and
circular-free.

Public APIs:
    - `Backbone`: Abstract base class for backbone feature extractors.
    - `UNetBackboneConfig`: Protocol for UNet backbone configuration.
    - `build_unet_backbone`: Build a configured UNet backbone instance.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'Backbone',
    'UNetBackboneConfig',
    'build_unet_backbone',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import Backbone
    from .factory import (
        UNetBackboneConfig,
        build_unet_backbone,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.base': ['Backbone'],
    '.factory': [
        'UNetBackboneConfig',
        'build_unet_backbone',
    ],
})
