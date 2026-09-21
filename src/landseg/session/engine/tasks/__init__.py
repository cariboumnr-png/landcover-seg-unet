# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Top-level namespace for `landseg.session.engine.runtime.tasks`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'CompositeLoss',
    'ConfusionMatrix',
    'ConsistencyRegularizer',
    'EngineTasks',
    'HeadSpec',
    'MTLMetricsAggregator',
    # functions
    'build_engine_tasks',
    # typing
    'TaskConfigShape',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import (
        EngineTasks,
        TaskConfigShape,
        build_engine_tasks,
    )
    from .heads import (
        HeadSpec,
    )
    from .loss.composite import (
        CompositeLoss,
    )
    from .metrics import (
        ConfusionMatrix,
        MTLMetricsAggregator,
    )
    from .regularization import (
        ConsistencyRegularizer,
    )


def __getattr__(name: str):
    if name in {
        'EngineTasks',
        'TaskConfigShape',
        'build_engine_tasks',
    }:
        obj = importlib.import_module('.builder', __package__)
        return getattr(obj, name)

    if name in {'HeadSpec'}:
        obj = importlib.import_module('.heads', __package__)
        return getattr(obj, name)

    if name in {'CompositeLoss'}:
        obj = importlib.import_module('.loss.composite', __package__)
        return getattr(obj, name)

    if name in {
        'ConfusionMatrix',
        'MTLMetricsAggregator',
    }:
        obj = importlib.import_module('.metrics', __package__)
        return getattr(obj, name)

    if name in {'ConsistencyRegularizer'}:
        obj = importlib.import_module('.regularization', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
