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
    # typing
]

# for static check
if typing.TYPE_CHECKING:
    from .default import default_action
    from .evaluate_model import evaluate
    from .ingest_data import ingest
    from .prepare_data import prepare
    from .train_model import train
    from .train_overfit import overfit

def __getattr__(name: str):

    if name in {'default_action'}:
        return getattr(importlib.import_module('.default', __package__), name)

    if name in {'evaluate'}:
        return getattr(importlib.import_module('.evaluate_model', __package__), name)

    if name in {'ingest'}:
        return getattr(importlib.import_module('.ingest_data', __package__), name)

    if name in {'prepare'}:
        return getattr(importlib.import_module('.prepare_data', __package__), name)

    if name in {'train'}:
        return getattr(importlib.import_module('.train_model', __package__), name)

    if name in {'overfit'}:
        return getattr(importlib.import_module('.train_overfit', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
