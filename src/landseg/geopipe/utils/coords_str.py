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
Coordinate and block name string conversion utilities.

This module provides helper functions to convert between integer
coordinate tuples (x, y) and canonical block filename identifiers.

Public APIs:
    - `xy_name`: convert (x, y) coords to canonical block name.
    - `name_xy`: convert canonical block name to (x, y) coords.
'''

# standard imports
from __future__ import annotations


# ----- public functions
def xy_name(coords: tuple[int, int]) -> str:
    '''
    Convert (x, y) coordinates to a canonical block name string.

    Args:
        coords:
            Tuple of integer pixel coordinates (x, y).

    Returns:
        str:
            Canonical block name formatted as 'row_YYYYYY_col_XXXXXX'.
    '''
    x, y = coords
    return f'row_{y:06d}_col_{x:06d}'


def name_xy(name: str) -> tuple[int, int]:
    '''
    Convert a canonical block name back to (x, y) coordinates.

    Args:
        name:
            Canonical block name formatted as 'row_YYYYYY_col_XXXXXX'.

    Returns:
        tuple[int, int]:
            Tuple of integer pixel coordinates (x, y).
    '''
    split = name.split('_')
    y_str, x_str = split[1], split[3]
    return int(x_str), int(y_str)
