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
Lightweight parallel execution framework supporting concurrency.

Compatibility shim delegating to `landseg.runtime.concurrency`.

Public APIs:
    - `ParallelExecutor`: Configurable process and thread task executor.
    - `get_default_worker_count`: Resolve default worker parallelism.
'''

# local imports
from landseg.runtime import concurrency

# re-exports
CORE_N = concurrency.CORE_N
ParallelExecutor = concurrency.ParallelExecutor
get_default_worker_count = concurrency.get_default_worker_count

__all__ = [
    'CORE_N',
    'ParallelExecutor',
    'get_default_worker_count',
]
