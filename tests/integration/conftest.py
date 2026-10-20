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
Shared configuration and fixtures for integration tests.
'''

# standard imports
import pathlib
# third-party imports
import pytest
# local imports
import landseg.testing as testing

# absolute path to repository root
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent


# ----- session fixtures
@pytest.fixture(scope='session', autouse=True)
def ensure_canonical_dummy_data():
    '''
    Ensure canonical on-disk dummy dataset exists before integration suite.

    Equivalent to running `python scripts/generate_dummy_data.py --yes`.
    '''
    in_dir = ROOT_DIR / 'experiment' / 'input'
    paths = testing.PreHarmonizationDummyPaths(root=in_dir)
    if not paths.all_paths_exist:
        testing.generate_pre_harmonization_data(in_dir)
    return paths
