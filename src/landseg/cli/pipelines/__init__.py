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
Top-level namespace for `landseg.cli.pipelines`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'default_action',
    'evaluate',
    'ingest',
    'overfit',
    'prepare',
    'train',
    'trial'
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .default import default_action
    from .data_ingest import ingest
    from .data_prepare import prepare
    from .diagnose_overfit import overfit
    from .model_evaluate import evaluate
    from .model_train import train
    from .study_trial import trial

def __getattr__(name: str):

    if name in {'default_action'}:
        return getattr(importlib.import_module('.default', __package__), name)

    if name in {'overfit'}:
        return getattr(importlib.import_module('.diagnose_overfit', __package__), name)

    if name in {'ingest'}:
        return getattr(importlib.import_module('.data_ingest', __package__), name)

    if name in {'prepare'}:
        return getattr(importlib.import_module('.data_prepare', __package__), name)

    if name in {'evaluate'}:
        return getattr(importlib.import_module('.model_evaluate', __package__), name)

    if name in {'train'}:
        return getattr(importlib.import_module('.model_train', __package__), name)

    if name in {'trial'}:
        return getattr(importlib.import_module('.study_trial', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
