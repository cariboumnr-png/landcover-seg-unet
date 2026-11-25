# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Unit tests for lazy resolution on top-level `landseg.adapters`.'''

# standard imports
import subprocess
import sys


# ----- lazy isolation tests
def test_importing_adapters_api_does_not_load_cli_or_hydra():
    '''
    Given: A fresh Python interpreter.
    When: Importing `landseg.adapters.api` directly or via `adapters.api`.
    Then: Neither `landseg.adapters.cli` nor `hydra` is imported.
    '''
    probe = (
        'import sys; '
        'import landseg.adapters.api as api; '
        'assert hasattr(api, "run"); '
        'assert "landseg.adapters.cli" not in sys.modules; '
        'assert "hydra" not in sys.modules'
    )
    res = subprocess.run(
        [sys.executable, '-c', probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr


def test_importing_adapters_cli_does_not_load_api():
    '''
    Given: A fresh Python interpreter.
    When: Importing `landseg.adapters.cli`.
    Then: `landseg.adapters.api` is not imported eagerly.
    '''
    probe = (
        'import sys; '
        'import landseg.adapters.cli as cli; '
        'assert hasattr(cli, "main"); '
        'assert "landseg.adapters.api" not in sys.modules'
    )
    res = subprocess.run(
        [sys.executable, '-c', probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
