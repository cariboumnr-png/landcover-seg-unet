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
Top-level namespace for `landseg.core`.

Exposes core data specification contracts, model protocols, and session
step result containers via lazy resolution to keep import order simple
and circular-free.

Public APIs:
    - `AccumulatedMetrics`: Container for IoU metrics and active views.
    - `DataSpecs`: Canonical dataset specs for trainers and models.
    - `Domains`: Discrete and continuous domain metadata container.
    - `Heads`: Head-wise label statistics and topology container.
    - `InferStepResults`: Container for epoch inference step results.
    - `Meta`: General dataset metadata for image and label specs.
    - `MultiheadModelLike`: Structural protocol for multihead models.
    - `SessionStepResults`: Immutable container for epoch step metrics.
    - `SessionStepSummary`: Immutable training progress snapshot.
    - `Splits`: Train, validation, and test block file mappings.
    - `TrainStepResults`: Container for training step loss metrics.
    - `ValStepResults`: Container for aggregated validation metrics.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'AccumulatedMetrics',
    'DataSpecs',
    'Domains',
    'Heads',
    'InferStepResults',
    'Meta',
    'MultiheadModelLike',
    'SessionStepResults',
    'SessionStepSummary',
    'Splits',
    'TrainStepResults',
    'ValStepResults',
]


# for static check
if typing.TYPE_CHECKING:
    from .data_specs import (
        DataSpecs,
        Domains,
        Heads,
        Meta,
        Splits,
    )
    from .model_protocol import MultiheadModelLike
    from .session_results import (
        AccumulatedMetrics,
        InferStepResults,
        SessionStepResults,
        SessionStepSummary,
        TrainStepResults,
        ValStepResults,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.data_specs': [
        'DataSpecs',
        'Domains',
        'Heads',
        'Meta',
        'Splits',
    ],
    '.model_protocol': ['MultiheadModelLike'],
    '.session_results': [
        'AccumulatedMetrics',
        'InferStepResults',
        'SessionStepResults',
        'SessionStepSummary',
        'TrainStepResults',
        'ValStepResults',
    ],
})
