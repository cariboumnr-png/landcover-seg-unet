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
Top-level namespace for `landseg.geopipe.prepare`.

Exposes public dataset context builders, partitioners, materializers,
preparation schema, and schema generators via lazy resolution.

Public APIs:
    - DataBlocksView: in-memory manifest view for catalog blocks.
    - DatasetContext: unified immutable dataset preparation context.
    - PartitionParameters: configuration for data block partitioning.
    - PreparationLogger: specialized logger for preparation runs.
    - PreparedSchema: TypedDict for dataset preparation schema.
    - build_dataset_context: construct full dataset preparation context.
    - build_schema: generate dataset preparation schema JSON.
    - run_datablocks_partition: partition data blocks into train/val/test.
    - run_materialize_blocks: orchestrate stats and block materialization.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'DataBlocksView',
    'DatasetContext',
    'PartitionParameters',
    'PreparationLogger',
    # functions
    'build_dataset_context',
    'build_schema',
    'run_datablocks_partition',
    'run_materialize_blocks',
    # types
    'PreparedSchema',
]


# for static check
if typing.TYPE_CHECKING:
    from .common import PreparationLogger, PreparedSchema
    from .data_context import (
        DataBlocksView,
        DatasetContext,
        build_dataset_context,
    )
    from .data_partition import PartitionParameters, run_datablocks_partition
    from .materialize_blocks import run_materialize_blocks
    from .schema import build_schema


def __getattr__(name: str):

    if name in {
        'DataBlocksView',
        'DatasetContext',
        'build_dataset_context',
    }:
        return getattr(
            importlib.import_module('.data_context', __package__), name
        )

    if name in {'PreparationLogger', 'PreparedSchema'}:
        return getattr(importlib.import_module('.common', __package__), name)

    if name in {'PartitionParameters', 'run_datablocks_partition'}:
        return getattr(
            importlib.import_module('.data_partition', __package__), name
        )

    if name in {'run_materialize_blocks'}:
        return getattr(
            importlib.import_module('.materialize_blocks', __package__), name
        )

    if name in {'build_schema'}:
        return getattr(importlib.import_module('.schema', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
