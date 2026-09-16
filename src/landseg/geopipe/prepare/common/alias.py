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
Type aliases for dataset preparation workflows.

Exposes array and coordinate type aliases used across the geopipe.prepare
pipeline modules for static type checking.

Public APIs:
    - IntArray: generic numpy integer array alias.
    - Int64Array: int64 numpy array alias.
    - Float32Array: float32 numpy array alias.
    - Float64Array: float64 numpy array alias.
    - MaskArray: boolean mask numpy array alias.
    - CoordsList: list of integer coordinate tuples.
'''

# standard imports
import typing
# third-party imports
import numpy.typing


# ----- typing aliases
IntArray: typing.TypeAlias = numpy.typing.NDArray[numpy.integer]
'''A generic `numpy` integer array.'''

Int64Array: typing.TypeAlias = numpy.typing.NDArray[numpy.int64]
'''A generic `numpy` array with `Int64` dtype.'''

Float32Array: typing.TypeAlias = numpy.typing.NDArray[numpy.float32]
'''A generic `numpy` array with `Float32` dtype.'''

Float64Array: typing.TypeAlias = numpy.typing.NDArray[numpy.float64]
'''A generic `numpy` array with `Float64` dtype.'''

MaskArray: typing.TypeAlias = numpy.typing.NDArray[numpy.bool]
'''A generic `numpy` array with `bool` dtype.'''

# coordinates
CoordsList: typing.TypeAlias = list[tuple[int, int]]
'''A simple list of tuples of two integers represent coordinates.'''
