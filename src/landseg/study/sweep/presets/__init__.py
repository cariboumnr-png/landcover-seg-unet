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

# pylint: disable=too-many-return-statements

'''
Top-level namespace for `landseg.study.sweep.objective_presets`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    # functions
    'obj_base',
    'obj_optimizer',
    'obj_throughput',
    'obj_data_geometry',
    'obj_context_window',
    'obj_architecture',
    'obj_bottleneck',
    'obj_conditioning',
    'obj_loss_balance',
    'obj_loss_aux',
    'obj_regularization',
    'obj_head_weights',
    'obj_mtl_joint',
    'obj_hierarchy',
    'comp_quick',
    'comp_capacity',
    'comp_mtl_quality',
    'comp_candidate',
    'resolve'
    # types
]

# for static check
if typing.TYPE_CHECKING:
    from ._registry import resolve
    from .base import obj_base
    from .optimizer import obj_optimizer, obj_throughput
    from .data import obj_data_geometry, obj_context_window
    from .architecture import obj_architecture, obj_bottleneck, obj_conditioning
    from .losses import obj_loss_balance, obj_loss_aux, obj_regularization
    from .multitask import obj_head_weights, obj_mtl_joint, obj_hierarchy
    from .composite import comp_quick, comp_capacity, comp_mtl_quality, comp_candidate

def __getattr__(name: str):

    if name in {'resolve'}:
        return getattr(importlib.import_module('._registry', __package__), name)

    if name in {'obj_base'}:
        return getattr(importlib.import_module('.base', __package__), name)

    if name in {'obj_optimizer', 'obj_throughput'}:
        return getattr(importlib.import_module('.optimizer', __package__), name)

    if name in {'obj_data_geometry', 'obj_context_window'}:
        return getattr(importlib.import_module('.data', __package__), name)

    if name in {'obj_architecture', 'obj_bottleneck', 'obj_conditioning'}:
        return getattr(importlib.import_module('.architecture', __package__), name)

    if name in {'obj_loss_balance', 'obj_loss_aux', 'obj_regularization'}:
        return getattr(importlib.import_module('.losses', __package__), name)

    if name in {'obj_head_weights', 'obj_mtl_joint', 'obj_hierarchy'}:
        return getattr(importlib.import_module('.multitask', __package__), name)

    if name in {'comp_quick', 'comp_capacity', 'comp_mtl_quality', 'comp_candidate'}:
        return getattr(importlib.import_module('.composite', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
