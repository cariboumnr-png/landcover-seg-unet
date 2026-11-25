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
Top-level namespace for `landseg.execution.preflight`.

Provides unified, non-destructive pre-flight diagnostic validation
routines, probe schemas, and terminal status dashboard generators.

Public APIs:
    - `PreflightResult`: Aggregated pre-flight validation result record.
    - `assert_target_ready`: Fail-fast readiness gate for a target.
    - `inspect_target`: Run pre-flight probes for a single target.
    - `run_preflight`: Dispatch pre-flight validation across targets.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'PreflightResult',
    'assert_target_ready',
    'inspect_target',
    'run_preflight',
]


# for static check
if typing.TYPE_CHECKING:
    from .engine import (
        assert_target_ready,
        inspect_target,
        run_preflight,
    )
    from .schema import PreflightResult


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.engine': [
        'assert_target_ready',
        'inspect_target',
        'run_preflight',
    ],
    '.schema': ['PreflightResult'],
})
