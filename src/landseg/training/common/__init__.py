# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Top-level namespace for `landseg.training.common`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    # types
    'CallBacksLike',
    'CompositeLossLike',
    'DataLoadersLike',
    'HeadLossesLike',
    'HeadMetricsLike',
    'HeadSpecsLike',
    'InferCallbackLike',
    'LoggingCallbackLike',
    'MetricLike',
    'MultiheadModelLike',
    'OptimizationLike',
    'ProgressCallbackLike',
    'RuntimeConfigLike',
    'RuntimeStateLike',
    'SpecLike',
    'TrainerCallbackLike',
    'TrainerComponentsLike',
    'TrainerLike',
    'ValCallbackLike',
]
# for static check
if typing.TYPE_CHECKING:
    from .trainer import TrainerLike
    from .trainer_comps import (
        CallBacksLike,
        CompositeLossLike,
        DataLoadersLike,
        HeadLossesLike,
        HeadMetricsLike,
        HeadSpecsLike,
        InferCallbackLike,
        LoggingCallbackLike,
        MetricLike,
        MultiheadModelLike,
        OptimizationLike,
        ProgressCallbackLike,
        SpecLike,
        TrainerCallbackLike,
        TrainerComponentsLike,
        ValCallbackLike,
    )
    from .trainer_config import RuntimeConfigLike
    from .trainer_state import RuntimeStateLike

def __getattr__(name: str):

    if name in ['TrainerLike']:
        return getattr(importlib.import_module('.trainer', __package__), name)
    if name in [
        'CallBacksLike',
        'CompositeLossLike',
        'DataLoadersLike',
        'HeadSpecsLike',
        'HeadLossesLike',
        'HeadMetricsLike',
        'InferCallbackLike',
        'LoggingCallbackLike',
        'MetricLike',
        'MultiheadModelLike',
        'OptimizationLike',
        'ProgressCallbackLike',
        'SpecLike',
        'TrainerCallbackLike',
        'TrainerComponentsLike',
        'ValCallbackLike'
    ]:
        return getattr(importlib.import_module('.trainer_comps', __package__), name)
    if name in ['RuntimeConfigLike']:
        return getattr(importlib.import_module('.trainer_config', __package__), name)
    if name in ['RuntimeStateLike']:
        return getattr(importlib.import_module('.trainer_state', __package__), name)

    raise AttributeError(name)
