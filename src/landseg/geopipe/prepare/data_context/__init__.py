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

'''
Top-level namespace for `landseg.geopipe.prepare.data_context`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.

Public APIs:
    - DataBlocksView: in-memory manifest view for catalog blocks.
    - DatasetContext: unified immutable dataset preparation context.
    - FeatureSelection: selected band names and channel indices.
    - TargetHeadsContext: resolved multi-head target hierarchy.
    - build_dataset_context: construct full preparation context.
    - derive_head_class_counts: derive counts across all target heads.
    - read_catalog: load dataset catalog into DataBlocksView.
    - resolve_feature_channels: resolve active input feature channels.
    - resolve_focal_head: resolve focal target head for partitioning.
    - resolve_target_heads: resolve multi-head hierarchy and reclass.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataBlocksView',
    'DatasetContext',
    'FeatureSelection',
    'TargetHeadsContext',
    # functions
    'build_dataset_context',
    'derive_head_class_counts',
    'read_catalog',
    'resolve_feature_channels',
    'resolve_focal_head',
    'resolve_target_heads',
    # types
]


# for static check
if typing.TYPE_CHECKING:
    from .catalog import DataBlocksView, read_catalog
    from .context import DatasetContext, build_dataset_context
    from .semantics import (
        FeatureSelection,
        TargetHeadsContext,
        derive_head_class_counts,
        resolve_feature_channels,
        resolve_focal_head,
        resolve_target_heads,
    )


def __getattr__(name: str):

    if name in {'DataBlocksView', 'read_catalog'}:
        return getattr(importlib.import_module('.catalog', __package__), name)

    if name in {'DatasetContext', 'build_dataset_context'}:
        return getattr(importlib.import_module('.context', __package__), name)

    if name in {
        'FeatureSelection',
        'TargetHeadsContext',
        'derive_head_class_counts',
        'resolve_feature_channels',
        'resolve_focal_head',
        'resolve_target_heads',
    }:
        return getattr(
            importlib.import_module('.semantics', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
