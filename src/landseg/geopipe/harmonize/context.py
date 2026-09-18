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
Execution context resolution for raster harmonization.

Provides containers and loaders to resolve the canonical spatial grid
reference for data harmonization from upstream pipeline artifacts.

Public APIs:
    - `HarmonizationContext`: Container holding resolved world grid.
    - `build_harmonization_context`: Load grid context from report.
'''

# standard imports
from __future__ import annotations
import dataclasses
import os
# local imports
import landseg.geopipe.core as geo_core


# ----- public dataclasses
@dataclasses.dataclass(frozen=True)
class HarmonizationContext:
    '''Resolved world grid reference container for harmonization.'''
    grid: geo_core.GridLayout
    grid_id: str
    grid_fpath: str


# ----- public functions
def build_harmonization_context(
    grid_source: str,
) -> HarmonizationContext:
    '''
    Load world grid reference context from an upstream grid report.

    Args:
        grid_source:
            File path to grid report JSON or directory containing it.

    Returns:
        HarmonizationContext:
            Execution context containing the restored GridLayout and
            grid reference metadata.
    '''
    if os.path.isdir(grid_source):
        report_fpath = geo_core.get_grid_report_fpath(grid_source)
    else:
        report_fpath = grid_source
    grid_report = geo_core.read_grid_report(report_fpath)
    world_grid = geo_core.load_grid_from_fpath(grid_report['grid_fpath'])
    return HarmonizationContext(
        grid=world_grid,
        grid_id=grid_report['grid_id'],
        grid_fpath=grid_report['grid_fpath'],
    )
