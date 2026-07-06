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
Utility script to generate dummy/mock geospatial datasets (GeoTIFFs)
and corresponding configurations for local pipeline runs and testing.
'''

# standard imports
import os
import sys
# local imports
import landseg.testing as testing

# -------------------------------Main Executable-------------------------------
if __name__ == '__main__':

    # check if the directory already exists and is not empty
    DEFAULT_DIR = './experiment/input'
    M = 'Generating dummy data will overwrite existing files. Proceed? [y/N]: '
    if os.path.exists(DEFAULT_DIR) and os.listdir(DEFAULT_DIR):
        print(
            f'WARNING: Target directory "{DEFAULT_DIR}" '
            f'already exists and is not empty.'
        )
        response = input(M)
        if response.strip().lower() not in ('y', 'yes'):
            print('Aborted.')
            sys.exit(0)
    testing.generate_dummy_data(DEFAULT_DIR)
