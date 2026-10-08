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
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # types
    'PreflightResult',
    'ProbeResult',
    'ProbeStatus',
    # functions
    'assert_target_prerequisites',
    'export_preflight_report',
    'format_preflight_report',
    'inspect_target',
    'run_preflight',
]


# for static check
if typing.TYPE_CHECKING:
    from .engine import (
        inspect_target,
        run_preflight,
    )
    from .prerequisites import (
        assert_target_prerequisites
    )
    from .reporter import (
        export_preflight_report,
        format_preflight_report,
    )
    from .schema import (
        PreflightResult,
        ProbeResult,
        ProbeStatus,
    )


def __getattr__(name: str):

    if name in {'assert_target_prerequisites'}:
        obj = importlib.import_module('.prerequisites', __package__)
        return getattr(obj, name)

    if name in {'PreflightResult', 'ProbeResult', 'ProbeStatus'}:
        obj = importlib.import_module('.schema', __package__)
        return getattr(obj, name)

    if name in {'inspect_target', 'run_preflight'}:
        obj = importlib.import_module('.engine', __package__)
        return getattr(obj, name)

    if name in {'export_preflight_report', 'format_preflight_report'}:
        obj = importlib.import_module('.reporter', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
