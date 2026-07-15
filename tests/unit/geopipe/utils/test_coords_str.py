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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for coordinates-to-string format utilities (coords_str.py).'''

# local imports
import landseg.geopipe.utils.coords_str as coords_str


# ----- `xy_name` tests
def test_xy_name():
    '''
    Given: A tuple of (x, y) coordinates.
    When: Converting to a canonical string name.
    Then: Formats as 'row_YYYYYY_col_XXXXXX' with zero-padding.
    '''
    assert coords_str.xy_name((12, 34)) == 'row_000034_col_000012'
    assert coords_str.xy_name((0, 0)) == 'row_000000_col_000000'


# ----- `name_xy` tests
def test_name_xy():
    '''
    Given: A canonical row/col block name string.
    When: Converting back to coordinates.
    Then: Correctly parses out the (x, y) integer tuple.
    '''
    assert coords_str.name_xy('row_000034_col_000012') == (12, 34)
    assert coords_str.name_xy('row_000000_col_000000') == (0, 0)
