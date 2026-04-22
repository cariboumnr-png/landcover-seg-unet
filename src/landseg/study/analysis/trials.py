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

'''
Trial analysis.
'''

# standard imports
import typing
# third-party imports
import optuna
import optuna.trial

def rank_trials(
    study_name: str,
    storage: str,
    *,
    top_k: int | None = None,
    ascending: bool = True
) -> list[dict[str, typing.Any]]:
    '''
    Load a study, rank completed trials, and return results as list.

    Args:
        study_name: Name of the study.
        storage: Optuna storage URL (e.g., 'sqlite:///example.db').
        top_k: If set, limit to top_k trials.
        ascending: True for minimization (lower is better), False for
            maximization.

    Returns:
        List of dicts ready for JSON serialization.
    '''

    def trial_value(t) -> float:
        assert t.value is not None
        return t.value

    study = optuna.load_study(study_name=study_name, storage=storage)

    # Filter completed trials
    trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]

    # Sort trials by objective value
    trials_sorted = sorted(trials, key=trial_value, reverse=not ascending)

    if top_k is not None:
        trials_sorted = trials_sorted[:top_k]

    # Serialize results
    results: list[dict[str, typing.Any]] = []
    for rank, t in enumerate(trials_sorted, start=1):
        results.append({
            'rank': rank,
            'trial_number': t.number,
            'value': t.value,
            'params': t.params,
            'datetime_start': t.datetime_start.isoformat() if t.datetime_start else None,
            'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None,
        })
    return results
