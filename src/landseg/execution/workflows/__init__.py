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

# pylint: disable=too-many-return-statements

'''
Top-level namespace for `landseg.execution.workflows`.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # functions
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



def __getattr__(name: str):
    if name in {'execute_batch_ingest'}:
        obj = importlib.import_module('.batch_ingest', __package__)
        return getattr(obj, name)

    if name in {'execute_default_action'}:
        obj = importlib.import_module('.default_action', __package__)
        return getattr(obj, name)

    if name in {'execute_diagnose_overfit'}:
        obj = importlib.import_module('.diagnose_overfit', __package__)
        return getattr(obj, name)

    if name in {'execute_e2e_experiment'}:
        obj = importlib.import_module('.e2e_experiment', __package__)
        return getattr(obj, name)

    if name in {'execute_e2e_intake'}:
        obj = importlib.import_module('.e2e_intake', __package__)
        return getattr(obj, name)

    if name in {'execute_study_analysis'}:
        obj = importlib.import_module('.study_analysis', __package__)
        return getattr(obj, name)

    if name in {'execute_study_sweep'}:
        obj = importlib.import_module('.study_sweep', __package__)
        return getattr(obj, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
