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

Compatibility shim delegating to `landseg.runtime.platform`.

Public APIs:
    - `get_cluster_id`: Probe active compute cluster identifier.
    - `get_dbutils`: Locate active dbutils instance.
    - `get_git_commit`: Retrieve git commit SHA of active repository.
    - `get_runtime_version`: Probe Databricks runtime version.
    - `is_databricks`: Probe whether runtime is inside Databricks.
'''

# local imports
from landseg.runtime import platform

# re-exports
get_cluster_id = platform.get_cluster_id
get_dbutils = platform.get_dbutils
get_git_commit = platform.get_git_commit
get_runtime_version = platform.get_runtime_version
is_databricks = platform.is_databricks

__all__ = [
    'get_cluster_id',
    'get_dbutils',
    'get_git_commit',
    'get_runtime_version',
    'is_databricks',
]
