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

'''Enumeration of artifact lifecycle policies.'''

# standard imports
import enum

class LifecyclePolicy(enum.Enum):
    '''
    Artifacts lifecycle policies.

    - `LOAD_ONLY`:        Use existing artifact only; do not build.
    - `LOAD_OR_FAIL`:     Load artifact if it exists, otherwise raise an error.
    - `BUILD_IF_MISSING`: Build artifact only if it does not already exist.
    - `REBUILD`:          Always rebuild artifact, ignoring any existing ones.
    - `REBUILD_IF_STALE`: Rebuild artifact only if it is outdated or invalid.
    '''

    LOAD_ONLY = enum.auto()
    LOAD_OR_FAIL = enum.auto()
    BUILD_IF_MISSING = enum.auto()
    REBUILD = enum.auto()
    REBUILD_IF_STALE = enum.auto()
