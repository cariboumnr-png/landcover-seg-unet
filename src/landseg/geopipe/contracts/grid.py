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

# pylint: disable=missing-function-docstring

'''
TypedDict definitions for world grid execution summaries and contracts.

This module provides schemas for serializing world grid metadata and
pipeline run reports.

Public APIs:
    = `WorldGridPrepConfig`: Protocol for pipeline configs.
    - `WorldGridReport`: TypedDict for world grid summary report.
    - `GridReportSchema`: TypedDict for overall grid pipeline report.
'''

# standard imports
from __future__ import annotations
import typing


# ----- public types
class WorldGridPrepConfig(typing.Protocol):
    '''Config shape to prepare world grid artifacts.'''
    @property
    def mode(self) -> str: ...
    @property
    def params(self) -> _GridParameters: ...
    @property
    def output_dpath(self) -> str: ...


class _GridParameters(typing.Protocol):
    @property
    def tile_size(self) -> tuple[int, int]: ...
    @property
    def tile_stride(self) -> tuple[int, int]: ...
    @property
    def ref_fpath(self) -> str | None: ...
    @property
    def crs_string(self) -> str | None: ...
    @property
    def origin(self) -> tuple[float, float] | None: ...
    @property
    def pixel_size(self) -> tuple[float, float] | None: ...
    @property
    def extent_in_crs_units(self) -> tuple[float, float] | None: ...


class WorldGridReport(typing.TypedDict):
    '''Summary report for a generated world grid layout.'''
    grid_fpath: str
    grid_id: str
    crs: str
    pixel_size: tuple[float, float]
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]


class GridReportSchema(typing.TypedDict):
    '''Execution report for the world grid pipeline run.'''
    run_id: str
    timestamp: str
    status: typing.Literal['SUCCESS', 'FAILED']
    grid: WorldGridReport
    total_tiles: int
