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

'''Unit tests for study sweep Optuna objective builder.'''

# standard imports
import pathlib
import types
# third-party imports
import omegaconf
import optuna
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.study.sweep.objectives as objectives_mod


def test_make_objective_records_trial_run_folder(tmp_path):
    '''
    Given: A runner builder yielding a step in an existing trial run dir.
    When: Running Optuna `study.optimize` over `make_objective`.
    Then: Persist `trial.user_attrs['run_folder']` pointing to that dir.
    '''
    trial_run = tmp_path / 'results' / 'run_0001'
    trial_run.mkdir(parents=True)
    step_results_fp = trial_run / 'step_results.json'

    fake_step = types.SimpleNamespace(
        val_metrics_value=0.77,
        epoch_in_phase=1,
        as_dict={'epoch_in_phase': 1},
    )

    def builder(_cfg):
        def runner():
            yield fake_step
        return step_results_fp, runner

    cfg = omegaconf.OmegaConf.to_object(
        omegaconf.OmegaConf.structured(configs.RootConfig)
    )
    assert isinstance(cfg, configs.RootConfig)
    obj = objectives_mod.make_objective(builder, cfg)
    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=1)

    assert study.trials[0].user_attrs['run_folder'] == str(trial_run)
    assert step_results_fp.is_file()


def test_make_objective_skips_missing_run_folder(tmp_path):
    '''
    Given: A runner builder whose step_results parent is not an initialized
        run directory before completion.
    When: Running `make_objective` with a direct missing run directory.
    Then: Complete optimization cleanly without failing on missing folder.
    '''
    missing_parent = tmp_path / 'missing_parent'
    step_results_fp = missing_parent / 'step_results.json'

    class FakeTrial:
        '''Minimal Optuna-like trial stub without filesystem hooks.'''

        def __init__(self):
            self.number = 0
            self.params: dict[str, object] = {}
            self.user_attrs: dict[str, object] = {}
            self.reports: list[tuple[float, int]] = []

        def suggest_float(self, name, low, high, **_kw):
            '''Return midpoint float parameter.'''
            val = float((low + high) / 2.0)
            self.params[name] = val
            return val

        def suggest_int(self, name, low, high, **_kw):
            '''Return lower-bound int parameter.'''
            self.params[name] = int(low)
            return int(low)

        def suggest_categorical(self, name, choices):
            '''Return first categorical choice.'''
            val = list(choices)[0]
            self.params[name] = val
            return val

        def report(self, value: float, step: int):
            '''Record reported intermediate metric.'''
            self.reports.append((value, step))

        def should_prune(self) -> bool:
            '''Never prune synthetic trial.'''
            return False

        def set_user_attr(self, key: str, value: object):
            '''Store user attribute.'''
            self.user_attrs[key] = value

    def builder(_cfg):
        def runner():
            yield core.SessionStepSummary(
                phase_name='p',
                phase_index=0,
                phase_max_epoch=1,
                epoch_in_phase=1,
                global_epoch=1,
                is_phase_end=True,
                is_run_end=True,
                stop_reason=None,
                val_metrics_name='iou',
                val_metrics_value=0.66,
                best_value_so_far=0.66,
                best_epoch_so_far=1,
                is_best_epoch=True,
                raw_metrics=core.SessionStepResults(),
            )
        return step_results_fp, runner

    cfg = omegaconf.OmegaConf.to_object(
        omegaconf.OmegaConf.structured(configs.RootConfig)
    )
    assert isinstance(cfg, configs.RootConfig)
    obj = objectives_mod.make_objective(builder, cfg)
    trial = FakeTrial()
    # Before Controller creates parent, run_folder didn't exist; post-persist it does
    score = obj(trial)  # type: ignore[arg-type]
    assert score == 0.66
    assert trial.user_attrs['run_folder'] == str(missing_parent)
