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
MLOps runtime environment and compute cluster detection utilities.

Provides runtime detection probes for Databricks cloud clusters, active
dbutils handles, cluster identifiers, runtime versions, and git lineage.

Public APIs:
    - `get_cluster_id`: Probe active compute cluster identifier.
    - `get_dbutils`: Locate active dbutils instance.
    - `get_git_commit`: Retrieve git commit SHA of active repository.
    - `get_runtime_version`: Probe Databricks runtime version.
    - `is_databricks`: Probe whether runtime is inside Databricks.
'''

# standard imports
import os
import subprocess
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

    # Check IPython user namespace if IPython is loaded
    ipython_mod = sys.modules.get('IPython')
    if ipython_mod is not None and hasattr(ipython_mod, 'get_ipython'):
        try:
            shell = ipython_mod.get_ipython()
            if shell is not None and hasattr(shell, 'user_ns'):
                if 'dbutils' in shell.user_ns:
                    return shell.user_ns['dbutils']
        except (AttributeError, KeyError):
            pass

    return None


def get_cluster_id() -> str:
    '''
    Retrieve Databricks compute cluster ID or local environment identifier.

    Returns:
        str:
            Value of DB_CLUSTER_ID environment variable or
            'local-environment'.
    '''
    return os.environ.get('DB_CLUSTER_ID', 'local-environment')


def get_runtime_version() -> str:
    '''
    Retrieve Databricks runtime version string or fallback.

    Returns:
        str:
            Value of DATABRICKS_RUNTIME_VERSION environment variable or
            'local'.
    '''
    return os.environ.get('DATABRICKS_RUNTIME_VERSION', 'local')


def get_git_commit() -> str:
    '''
    Retrieve git commit SHA from environment variables or git repository.

    Checks standard CI/CD and Databricks environment variables:
    `GIT_COMMIT`, `DATABRICKS_GIT_COMMIT`, `GITHUB_SHA`. Falls back to
    executing `git rev-parse HEAD`.

    Returns:
        str:
            Resolved Git commit SHA or 'unknown'.
    '''
    env_sha = (
        os.environ.get('GIT_COMMIT') or
        os.environ.get('DATABRICKS_GIT_COMMIT') or
        os.environ.get('GITHUB_SHA')
    )
    if env_sha:
        return env_sha

    try:
        res = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if res.returncode == 0 and res.stdout:
            return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass

    return 'unknown'
