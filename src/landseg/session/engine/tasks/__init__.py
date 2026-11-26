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
Top-level namespace for `landseg.session.engine.tasks`.

Exposes selected public APIs via lazy resolution to keep import order
simple and circular-free.

Public APIs:
    - `CompositeLoss`: Weighted combination of primitive loss modules.
    - `ConfusionMatrix`: Incremental per-head confusion matrix and IoU.
    - `ConsistencyRegularizer`: Cross-head consistency penalty module.
    - `EngineTasks`: Container for session-level task components.
    - `HeadSpec`: Configuration dataclass for a single training head.
    - `MTLMetricsAggregator`: Multi-task geometric and rule metrics.
    - `TaskConfigShape`: Protocol for engine task configuration.
    - `build_engine_tasks`: Construct head specs, losses, and metrics.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'CompositeLoss',
    'ConfusionMatrix',
    'ConsistencyRegularizer',
    'EngineTasks',
    'HeadSpec',
    'MTLMetricsAggregator',
    'TaskConfigShape',
    'build_engine_tasks',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import (
        EngineTasks,
        TaskConfigShape,
        build_engine_tasks,
    )
    from .heads import HeadSpec
    from .loss import CompositeLoss
    from .metrics import (
        ConfusionMatrix,
        MTLMetricsAggregator,
    )
    from .regularization import ConsistencyRegularizer


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.builder': [
        'EngineTasks',
        'TaskConfigShape',
        'build_engine_tasks',
    ],
    '.heads': ['HeadSpec'],
    '.loss': ['CompositeLoss'],
    '.metrics': [
        'ConfusionMatrix',
        'MTLMetricsAggregator',
    ],
    '.regularization': ['ConsistencyRegularizer'],
})
