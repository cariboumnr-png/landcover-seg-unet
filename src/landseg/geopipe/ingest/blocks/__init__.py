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
Top-level namespace for `landseg.geopipe.ingest.blocks`.

Coordinates data block construction, window mapping, and manifest
management through lazy module resolution.

Public APIs:
    - `BlockBuildingConfig`: Config container for block pipeline.
    - `BlockBuildingInputs`: I/O path inputs for block pipeline.
    - `BlockPipelineRuntimeContext`: Context for block pipeline.
    - `run_blocks_building`: Runs canonical data block pipeline.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'BlockBuildingConfig',
    'BlockBuildingInputs',
    'BlockPipelineRuntimeContext',
    'run_blocks_building',
]


# for static check
if typing.TYPE_CHECKING:
    from .assembler import (
        BlockBuildingConfig,
        BlockBuildingInputs,
    )
    from .pipeline import (
        BlockPipelineRuntimeContext,
        run_blocks_building,
    )


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.assembler': [
        'BlockBuildingConfig',
        'BlockBuildingInputs',
    ],
    '.pipeline': [
        'BlockPipelineRuntimeContext',
        'run_blocks_building',
    ],
})
