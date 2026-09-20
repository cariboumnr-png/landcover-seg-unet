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
Top-level namespace for `landseg.geopipe.prepare.partition`.

Exposes partition configuration parameters and dataset split runner
via lazy resolution.

Public APIs:
    - PartitionParameters: configuration parameters for partitioning.
    - run_datablocks_partition: partition blocks into train/val/test.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'PartitionParameters',
    # functions
    'run_datablocks_partition',
]


# for static check
if typing.TYPE_CHECKING:
    from .orchestration import (
        PartitionParameters,
    )
    from .runner import (
        run_datablocks_partition,
    )


def __getattr__(name: str):
    if name in {'PartitionParameters'}:
        obj = importlib.import_module('.orchestration', __package__)
        return getattr(obj, name)

    if name in {'run_datablocks_partition'}:
        obj = importlib.import_module('.runner', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
