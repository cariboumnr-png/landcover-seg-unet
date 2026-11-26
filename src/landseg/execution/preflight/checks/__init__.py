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
Pre-flight diagnostic check functions.

Each submodule groups checks by `ProbeCategory`. All checks accept an
`InspectionContext` (plus explicit paths where relevant) and return a
`ProbeResult`; none mutate pipeline state.

Public APIs:
    - `contracts`: Stage contract checks backed by `geopipe`.
    - `filesystem`: Output directory access and writability checks.
    - `hardware`: Accelerator availability and VRAM headroom checks.
    - `ledger`: Block pool and prepared-state ledger checks.
    - `lineage`: Upstream report, manifest, and artifact checks.
    - `model`: Model architecture, checkpoint, and split checks.
    - `policy`: Execution collision-handling policy checks.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'contracts',
    'filesystem',
    'hardware',
    'ledger',
    'lineage',
    'model',
    'policy',
]


# for static check
if typing.TYPE_CHECKING:
    from . import (
        contracts,
        filesystem,
        hardware,
        ledger,
        lineage,
        model,
        policy,
    )


__getattr__, __dir__ = _lazy.attach(
    __name__,
    {},
    submodules=[
        'contracts',
        'filesystem',
        'hardware',
        'ledger',
        'lineage',
        'model',
        'policy',
    ],
)
