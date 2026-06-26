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

'''
Bootstrapping runner script for Databricks job compute or VM environments.
Adds 'src' directory to python path and invokes the CLI entry point.
'''

# standard imports
import os
import sys
# local imports
from landseg.adapters.cli.cli import main

# Locate absolute path of workspace root and insert 'src' folder into path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_SRC_DIR = os.path.join(_WORKSPACE_ROOT, 'src')
sys.path.insert(0, _SRC_DIR)

if __name__ == '__main__':
    main()
