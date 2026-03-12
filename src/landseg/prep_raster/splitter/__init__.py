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
Top-level namespace for `landseg.dataset.splitter`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'count_label_class',
    'select_val_blocks',
    'score_blocks',
    'split_blocks',
    # typing
    'BlockScore',
    'ScoreParams',
]

# for static check
if typing.TYPE_CHECKING:
    from .counter import count_label_class
    from .score import BlockScore, ScoreParams, score_blocks
    from .selector import select_val_blocks
    from .splitter import split_blocks

def __getattr__(name: str):

    if name in ['count_label_class']:
        return getattr(importlib.import_module('.counter', __package__), name)
    if name in ['BlockScore', 'ScoreParams', 'score_blocks']:
        return getattr(importlib.import_module('.score', __package__), name)
    if name in ['select_val_blocks']:
        return getattr(importlib.import_module('.selector', __package__), name)
    if name in ['split_blocks']:
        return getattr(importlib.import_module('.splitter', __package__), name)

    raise AttributeError(name)
