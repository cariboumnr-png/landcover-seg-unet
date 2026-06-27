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

# pylint: disable=no-value-for-parameter
# pylint: disable=wrong-import-position

'''
Bootstrapping runner script for Databricks job compute or VM environments.
Adds 'src' directory to python path and invokes the CLI entry point.
'''

# standard imports
import os
import sys

def _get_script_dir() -> str:
    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()  # Databricks fallback

# Locate absolute path of workspace root and insert 'src' folder into path
_SCRIPT_DIR = _get_script_dir()
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_SRC_DIR = os.path.join(_WORKSPACE_ROOT, 'src')

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# local imports after sys path modification
from landseg.adapters.cli.cli import main

if __name__ == '__main__':
    main()
