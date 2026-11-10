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

'''
Notebook runtime environment detection utilities.

Provides runtime detection probes for interactive IPython notebook kernels
(Jupyter and Databricks notebook shells).

Public APIs:
    - `is_notebook`: Probe whether code is running inside a notebook.
'''

# standard imports
from __future__ import annotations
# third-party imports
try:
    import IPython
except ImportError:
    IPython = None


# ----- public functions
def is_notebook() -> bool:
    '''
    Probe whether executing within an interactive notebook kernel.

    Checks active IPython kernel shell types across Jupyter and
    Databricks.

    Returns:
        bool:
            True if executing in an interactive notebook kernel.
    '''
    if IPython is None:
        return False

    try:
        shell = IPython.get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in {'ZMQInteractiveShell', 'DatabricksShell'}
    except AttributeError:
        return False
