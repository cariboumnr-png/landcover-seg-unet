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

'''
Utility helper functions.
'''

# coords <-> name helpers
def xy_name(coords: tuple[int, int]) -> str:
    '''Convert (x, y) to a canonical block name string.'''

    # e.g., (12, 34) -> row_000034_col_000012
    x, y = coords
    return f'row_{y:06d}_col_{x:06d}'

def name_xy(name: str) -> tuple[int, int]:
    '''Convert a canonical block name back to (x, y).'''

    # e.g.,  row_000034_col_000012 -> (12, 34)
    split = name.split('_')
    y_str, x_str = split[1], split[3]
    return int(x_str), int(y_str)
