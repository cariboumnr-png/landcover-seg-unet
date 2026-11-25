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

# pylint: disable=import-outside-toplevel

'''
Top-level namespace for `landseg.geopipe.grid`.

Exposes world grid construction and lifecycle management APIs via lazy
resolution to keep import order simple and circular-free.

Public APIs:
    - `GridLogger`: Logger tracking world grid execution and report.
    - `prepare_world_grid`: Build or load persisted world grid artifact.
    - `validators`: Preflight validation namespace.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import landseg._lazy as _lazy


class _Validators: # pylint: disable=R0903

    @staticmethod
    def grid_specs(config: contracts.WorldGridPrepConfig):
        '''Validate grid generation configurations.'''
        import landseg.geopipe.grid.builder as builder
        return builder.compile_inputs(config)

validators = _Validators()


__all__ = [
    'GridLogger',
    'prepare_world_grid',
    'validators',
]


# for static check
if typing.TYPE_CHECKING:
    import landseg.geopipe.contracts as contracts

    from .lifecycle import prepare_world_grid
    from .logger import GridLogger


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.lifecycle': ['prepare_world_grid'],
    '.logger': ['GridLogger'],
})
