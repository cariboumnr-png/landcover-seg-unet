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

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'UNet',
    'UNetBackbone',
    'UNetBodyConfig',
    'UNetPP',
    'UNetPPP',
]


# for static check
if typing.TYPE_CHECKING:
    from .base import (
        UNetBackbone,
    )
    from .configs import (
        UNetBodyConfig,
    )
    from .unet import (
        UNet,
    )
    from .unetpp import (
        UNetPP,
    )
    from .unetppp import (
        UNetPPP,
    )


def __getattr__(name: str):
    if name in {'UNetBackbone'}:
        obj = importlib.import_module('.base', __package__)
        return getattr(obj, name)

    if name in {'UNetBodyConfig'}:
        obj = importlib.import_module('.configs', __package__)
        return getattr(obj, name)

    if name in {'UNet'}:
        obj = importlib.import_module('.unet', __package__)
        return getattr(obj, name)

    if name in {'UNetPP'}:
        obj = importlib.import_module('.unetpp', __package__)
        return getattr(obj, name)

    if name in {'UNetPPP'}:
        obj = importlib.import_module('.unetppp', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
