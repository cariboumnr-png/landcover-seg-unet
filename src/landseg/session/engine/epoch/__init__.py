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
Top-level namespace for `landseg.session.engine.epoch`.

Exposes selected public APIs via lazy resolution to keep import order
simple and circular-free.

Public APIs:
    - `EngineRuntime`: Container for batch engine, optimizer, and tasks.
    - `EpochRunner`: Wrapper running one epoch of training or eval.
    - `EpochRunnerContext`: External context for building epoch runner.
    - `MultiHeadEvaluator`: Epoch-level validation and inference policy.
    - `MultiHeadTrainer`: Epoch-level multi-head training policy.
    - `ScheduleConfigShape`: Protocol for epoch schedule frequencies.
    - `build_epoch_runner`: Construct configured `EpochRunner`.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'EngineRuntime',
    'EpochRunner',
    'EpochRunnerContext',
    'MultiHeadEvaluator',
    'MultiHeadTrainer',
    'ScheduleConfigShape',
    'build_epoch_runner',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import (
        EpochRunnerContext,
        ScheduleConfigShape,
        build_epoch_runner,
    )
    from .policy import (
        EngineRuntime,
        MultiHeadEvaluator,
        MultiHeadTrainer,
    )
    from .runner import EpochRunner


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.builder': [
        'EpochRunnerContext',
        'ScheduleConfigShape',
        'build_epoch_runner',
    ],
    '.policy': [
        'EngineRuntime',
        'MultiHeadEvaluator',
        'MultiHeadTrainer',
    ],
    '.runner': ['EpochRunner'],
})
