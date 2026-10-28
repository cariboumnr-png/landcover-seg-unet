# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

# pylint: disable=import-outside-toplevel

'''
Top-level namespace for `landseg.geopipe.prepare`.

Exposes public dataset context builders, partitioners, materializers,
preparation schema, handlers, and schema generators via lazy resolution.

Public APIs:
    - `PreparationContext`: Resolved execution context for prepare.
    - `PreparationLogger`: Specialized logger for preparation runs.
    - `build_preparation_context`: Build context from catalog & schema.
    - `run_data_preparation`: pipeline runner.
'''

# standard imports
from __future__ import annotations
import importlib
import typing


class _Validators:

    @staticmethod
    def dataset_view(*args, **kwargs):
        '''Validate and compile dataset view.'''
        import landseg.geopipe.prepare.dataset.view as view
        return view.build_dataset_view(*args, **kwargs)

    @staticmethod
    def feature_channels(*args, **kwargs):
        '''Validate requested feature channels.'''
        import landseg.geopipe.prepare.dataset.semantics as semantics
        return semantics.resolve_feature_channels(*args, **kwargs)

    @staticmethod
    def focal_head(*args, **kwargs):
        '''Validate focal target head selection.'''
        import landseg.geopipe.prepare.dataset.semantics as semantics
        return semantics.resolve_focal_head(*args, **kwargs)

    @staticmethod
    def preparation_context(*args, **kwargs):
        '''Validate preparation inputs.'''
        import landseg.geopipe.prepare.context as context
        return context.build_preparation_context(*args, **kwargs)

    @staticmethod
    def target_heads(*args, **kwargs):
        '''Validate multi-head target hierarchy.'''
        import landseg.geopipe.prepare.dataset.semantics as semantics
        return semantics.resolve_target_heads(*args, **kwargs)

validators = _Validators()


__all__ = [
    # classes
    'PreparationContext',
    'PreparationLogger',
    # functions
    'build_preparation_context',
    'run_data_preparation',
    # namespace
    'validators',
]


# for static check
if typing.TYPE_CHECKING:
    from .context import (
        PreparationContext,
        build_preparation_context,
    )
    from .logger import PreparationLogger
    from .pipeline import run_data_preparation


def __getattr__(name: str):

    if name in {
        'PreparationContext',
        'build_preparation_context'
    }:
        obj = importlib.import_module('.context', __package__)
        return getattr(obj, name)

    if name in {'PreparationLogger'}:
        obj = importlib.import_module('.logger', __package__)
        return getattr(obj, name)

    if name in {'run_data_preparation'}:
        obj = importlib.import_module('.pipeline', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
