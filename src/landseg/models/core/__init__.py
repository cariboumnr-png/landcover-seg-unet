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
Top-level namespace for `landseg.models.core`.

Exposes core domain conditioning adapters, routing utilities,
configuration contracts, multi-task head management, and numerical
safety helpers via lazy resolution to keep import order simple and
circular-free.

Public APIs:
    - `ConcatAdapter`: Channel concatenation domain conditioner.
    - `DomainContextRouter`: Router for domain conditioning payloads.
    - `DomainProjectionConfig`: TypedDict for domain vector projection.
    - `DomainTargetConfig`: Protocol for domain target configuration.
    - `DomainTargetPayload`: Container for routed domain tensors.
    - `FilmConditioner`: FiLM affine modulation domain conditioner.
    - `HeadManager`: Multi-task output head and freeze manager.
    - `NumericSafety`: AMP autocast and tensor clamping helper.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'ConcatAdapter',
    'DomainContextRouter',
    'DomainProjectionConfig',
    'DomainTargetConfig',
    'DomainTargetPayload',
    'FilmConditioner',
    'HeadManager',
    'NumericSafety',
]


# for static check
if typing.TYPE_CHECKING:
    from .conditioner import (
        ConcatAdapter,
        FilmConditioner,
    )
    from .config import (
        DomainProjectionConfig,
        DomainTargetConfig,
    )
    from .domains import (
        DomainContextRouter,
        DomainTargetPayload,
    )
    from .heads import HeadManager
    from .safety import NumericSafety


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.conditioner': [
        'ConcatAdapter',
        'FilmConditioner',
    ],
    '.config': [
        'DomainProjectionConfig',
        'DomainTargetConfig',
    ],
    '.domains': [
        'DomainContextRouter',
        'DomainTargetPayload',
    ],
    '.heads': ['HeadManager'],
    '.safety': ['NumericSafety'],
})
