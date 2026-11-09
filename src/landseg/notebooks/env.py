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

Provides runtime detection probes for Databricks cloud clusters and
interactive notebook kernels.

Public APIs:
    - `get_dbutils`: Locate active dbutils instance.
    - `is_databricks`: Probe whether runtime is inside Databricks.
    - `is_notebook`: Probe whether code is running inside a notebook.
'''

# standard imports
import os
import sys
import typing


# ----- public functions
def is_databricks() -> bool:
    '''
    Probe whether code is executing inside a Databricks environment.

    Checks standard Databricks runtime environment variable markers.

    Returns:
        bool:
            True if running in Databricks runtime, False otherwise.
    '''
    return (
        'DATABRICKS_RUNTIME_VERSION' in os.environ or
        'DB_CLUSTER_ID' in os.environ
    )


def is_notebook() -> bool:
    '''
    Probe whether executing within an interactive notebook kernel.

    Checks active IPython kernel shell types across Jupyter and
    Databricks.

    Returns:
        bool:
            True if executing in an interactive notebook kernel.
    '''
    try:
        import IPython  # pylint: disable=import-outside-toplevel
        shell = IPython.get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        return shell_name in {'ZMQInteractiveShell', 'DatabricksShell'}
    except (AttributeError, ImportError):
        return False


def get_dbutils(dbutils: typing.Any = None) -> typing.Any:
    '''
    Locate active dbutils instance from argument or caller scopes.

    Args:
        dbutils:
            Optional explicit dbutils handle.

    Returns:
        typing.Any:
            Active dbutils instance or None if unavailable.
    '''
    if dbutils is not None:
        return dbutils

    main_mod = sys.modules.get('__main__')
    if main_mod is not None and hasattr(main_mod, 'dbutils'):
        return getattr(main_mod, 'dbutils')

    try:
        import IPython  # pylint: disable=import-outside-toplevel
        shell = IPython.get_ipython()
        if shell is not None and hasattr(shell, 'user_ns'):
            if 'dbutils' in shell.user_ns:
                return shell.user_ns['dbutils']
    except (AttributeError, ImportError, KeyError):
        pass

    return None
