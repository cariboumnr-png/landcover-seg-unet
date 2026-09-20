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

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'AccumulatedMetrics',
    'DataSpecs',
    'Domains',
    'Heads',
    'InferStepResults',
    'Meta',
    'SessionStepResults',
    'SessionStepSummary',
    'Splits',
    'TrainStepResults',
    'ValStepResults',
    # typing
    'MultiheadModelLike',
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
    from .model_protocol import (
        MultiheadModelLike,
    )
    from .session_results import (
        AccumulatedMetrics,
        InferStepResults,
        SessionStepResults,
        SessionStepSummary,
        TrainStepResults,
        ValStepResults,
    )


def __getattr__(name: str):
    if name in {
        'DataSpecs',
        'Domains',
        'Heads',
        'Meta',
        'Splits',
    }:
        obj = importlib.import_module('.data_specs', __package__)
        return getattr(obj, name)

    if name in {'MultiheadModelLike'}:
        obj = importlib.import_module('.model_protocol', __package__)
        return getattr(obj, name)

    if name in {
        'AccumulatedMetrics',
        'InferStepResults',
        'SessionStepResults',
        'SessionStepSummary',
        'TrainStepResults',
        'ValStepResults',
    }:
        obj = importlib.import_module('.session_results', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
