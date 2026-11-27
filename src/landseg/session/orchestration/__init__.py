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
Top-level namespace for `landseg.session.orchestration`.

Exposes selected public symbols via lazy resolution to keep import
order simple and circular-free.

Public APIs:
    - `CheckpointCatalog`: Reader, writer, and resolver for session runs.
    - `ContinuousRunner`: Runner executing a single continuous phase.
    - `CurriculumRunner`: Runner executing sequential curriculum phases.
    - `OrchestrationConfigShape`: Protocol for orchestration configs.
    - `build_runner`: Construct a concrete orchestration runner.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'CheckpointCatalog',
    'ContinuousRunner',
    'CurriculumRunner',
    'OrchestrationConfigShape',
    'build_runner',
]


# for static check
if typing.TYPE_CHECKING:
    from .builder import build_runner
    from .catalog import CheckpointCatalog
    from .protocols import OrchestrationConfigShape
    from .runner import (
        ContinuousRunner,
        CurriculumRunner,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.builder': ['build_runner'],
    '.catalog': ['CheckpointCatalog'],
    '.protocols': ['OrchestrationConfigShape'],
    '.runner': [
        'ContinuousRunner',
        'CurriculumRunner',
    ],
})
