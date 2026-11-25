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
Top-level namespace for `landseg.geopipe.prepare.partition`.

Exposes partition configuration parameters and dataset split runner
via lazy resolution.

Public APIs:
    - `AOIConfig`: Configuration for AOI spatial boundaries.
    - `HydrationConfig`: Configuration for candidate hydration.
    - `PartitionConfig`: Configuration for data block partitioning.
    - `run_datablocks_partition`: Partition blocks into train/val/test.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'AOIConfig',
    'HydrationConfig',
    'PartitionConfig',
    'run_datablocks_partition',
]


# for static check
if typing.TYPE_CHECKING:
    from .orchestration import (
        AOIConfig,
        HydrationConfig,
        PartitionConfig,
    )
    from .runner import run_datablocks_partition


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.orchestration': [
        'AOIConfig',
        'HydrationConfig',
        'PartitionConfig',
    ],
    '.runner': ['run_datablocks_partition'],
})
