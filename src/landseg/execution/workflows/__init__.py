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
Top-level namespace for `landseg.execution.workflows`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
import typing
# local imports
import landseg._lazy as _lazy

__all__ = [
    'execute_batch_ingest',
    'execute_default_action',
    'execute_diagnose_overfit',
    'execute_e2e_experiment',
    'execute_e2e_intake',
    'execute_study_analysis',
    'execute_study_sweep',
]


# for static check
if typing.TYPE_CHECKING:
    from .batch_ingest import execute_batch_ingest
    from .default_action import execute_default_action
    from .diagnose_overfit import execute_diagnose_overfit
    from .e2e_experiment import execute_e2e_experiment
    from .e2e_intake import execute_e2e_intake
    from .study_analysis import execute_study_analysis
    from .study_sweep import execute_study_sweep


__getattr__, __dir__ = _lazy.attach(__name__, {
    '.batch_ingest': ['execute_batch_ingest'],
    '.default_action': ['execute_default_action'],
    '.diagnose_overfit': ['execute_diagnose_overfit'],
    '.e2e_experiment': ['execute_e2e_experiment'],
    '.e2e_intake': ['execute_e2e_intake'],
    '.study_analysis': ['execute_study_analysis'],
    '.study_sweep': ['execute_study_sweep'],
})
