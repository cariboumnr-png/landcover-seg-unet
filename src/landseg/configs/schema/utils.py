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

'''
Session schema utilities.
'''

# standard imports
import os
import typing

def file_exists(path: str) -> bool:
    return os.path.isfile(path) and os.path.exists(path)

def must_exist(path: str | None, tag: str) -> None:
    if path and not file_exists(path):
        raise FileNotFoundError(f'File [{tag}] is invalid: {path}')

def must_within(
    value: typing.Any,
    tag: str,
    mmin: int | float | None = None,
    mmax: int | float | None = None,
) -> None:
    if not isinstance(value, (int, float)):
        return
    rr = f'[{mmin}, {mmax}]'
    if mmin and value < mmin or mmax and value > mmax:
        raise ValueError(f'Value [{tag}] must be within {rr}, got: {value}')
