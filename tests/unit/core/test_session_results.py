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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for dataclasses and functions in session_results.py.'''

# third-party imports
import pytest
# local imports
import landseg.core as core
import landseg.core.session_results as session_results # for private access

# ----- SessionStepSummary
def test_session_step_summary_as_dict_contains_expected_keys():
    summary = core.SessionStepSummary(
        phase_name='train',
        phase_index=1,
        phase_max_epoch=5,
        epoch_in_phase=2,
        global_epoch=2,
        is_phase_end=False,
        is_run_end=False,
        stop_reason=None,
        val_metrics_name='IoU',
        val_metrics_value=0.5,
        best_value_so_far=0.5,
        best_epoch_so_far=2,
        is_best_epoch=True,
        raw_metrics=core.SessionStepResults(),
    )

    assert set(summary.as_dict) == {
        'phase_name',
        'phase_max_epoch',
        'epoch_in_phase',
        'validation_metrics_name',
        'validation_metrics_value',
        'training',
        'validation',
        'inference',
    }


def test_session_step_summary_as_dict_full():
    training = core.TrainStepResults(
        total_objective=1.5,
        head_losses={'loss': 0.4},
    )
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.6),
        }
    )
    inference = core.InferStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.7),
        }
    )
    raw = core.SessionStepResults(
        training=training,
        validation=validation,
        inference=inference,
    )

    summary = core.SessionStepSummary(
        phase_name='train',
        phase_index=0,
        phase_max_epoch=10,
        epoch_in_phase=3,
        global_epoch=3,
        is_phase_end=False,
        is_run_end=False,
        stop_reason=None,
        val_metrics_name='IoU from head1',
        val_metrics_value=0.6,
        best_value_so_far=0.6,
        best_epoch_so_far=3,
        is_best_epoch=True,
        raw_metrics=raw,
    )

    assert summary.as_dict == {
        'phase_name': 'train',
        'phase_max_epoch': 10,
        'epoch_in_phase': 3,
        'validation_metrics_name': 'IoU from head1',
        'validation_metrics_value': 0.6,
        'training': training.as_dict,
        'validation': validation.as_dict,
        'inference': inference.as_dict,
    }


def test_session_step_summary_as_dict_missing_results():
    summary = core.SessionStepSummary(
        phase_name='train',
        phase_index=0,
        phase_max_epoch=1,
        epoch_in_phase=1,
        global_epoch=1,
        is_phase_end=True,
        is_run_end=True,
        stop_reason='done',
        val_metrics_name='IoU',
        val_metrics_value=0.5,
        best_value_so_far=0.5,
        best_epoch_so_far=1,
        is_best_epoch=True,
        raw_metrics=core.SessionStepResults(),
    )

    assert summary.as_dict == {
        'phase_name': 'train',
        'phase_max_epoch': 1,
        'epoch_in_phase': 1,
        'validation_metrics_name': 'IoU',
        'validation_metrics_value': 0.5,
        'training': None,
        'validation': None,
        'inference': None,
    }


def test_session_step_summary_as_dict_returns_snapshot():
    training = core.TrainStepResults(
        total_objective=1.0,
        head_losses={'loss': 0.5},
    )
    raw = core.SessionStepResults(training=training)

    summary = core.SessionStepSummary(
        phase_name='train',
        phase_index=0,
        phase_max_epoch=1,
        epoch_in_phase=1,
        global_epoch=1,
        is_phase_end=True,
        is_run_end=True,
        stop_reason=None,
        val_metrics_name='IoU',
        val_metrics_value=0.5,
        best_value_so_far=0.5,
        best_epoch_so_far=1,
        is_best_epoch=True,
        raw_metrics=raw,
    )
    d = summary.as_dict
    training.head_losses['loss'] = 99.0

    assert d['training']['head_losses']['loss'] == 0.5


# ----- SessionStepResults
def test_session_step_results_target_objective_default():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.5),
        }
    )
    results = core.SessionStepResults(validation=validation)

    assert results.target_objective == 'IoU from head1'


def test_session_step_results_target_objective_weighted():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(),
            'head2': core.AccumulatedMetrics(),
        }
    )
    results = core.SessionStepResults(validation=validation)
    results.track('iou', {'head1': 0.25, 'head2': 0.75})

    assert results.target_objective == 'IoU = head1*0.25 + head2*0.75'


def test_session_step_results_target_objective_gem():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(),
            'head2': core.AccumulatedMetrics(),
        },
        mtl_metrics={
            'gem': 0.9,
        },
    )
    results = core.SessionStepResults(validation=validation)
    results.track('gem', None)

    assert results.target_objective == 'Global Exact Match over [head1 & head2]'


def test_session_step_results_target_objective_invalid_metric():
    validation = core.ValStepResults(
        head_metrics={
            'head': core.AccumulatedMetrics(),
        }
    )
    results = core.SessionStepResults(validation=validation)
    object.__setattr__(results, '_metric_name', 'invalid') # as in .track()

    with pytest.raises(ValueError, match='Invalid metric name'):
        _ = results.target_objective


def test_session_step_results_target_metrics_no_validation():
    results = core.SessionStepResults()

    assert results.target_metrics == -float('inf')


def test_session_step_results_target_metrics_default_first_head():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.6),
            'head2': core.AccumulatedMetrics(mean=0.9),
        }
    )
    results = core.SessionStepResults(validation=validation)

    assert results.target_metrics == pytest.approx(0.6)


def test_session_step_results_inference_metrics_no_inference():
    results = core.SessionStepResults()

    assert results.inference_metrics == -float('inf')


def test_session_step_results_inference_metrics_default_first_head():
    inference = core.InferStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.75),
            'head2': core.AccumulatedMetrics(mean=0.9),
        }
    )
    results = core.SessionStepResults(inference=inference)

    assert results.inference_metrics == pytest.approx(0.75)


def test_session_step_results_as_dict():
    results = core.SessionStepResults(
        training=core.TrainStepResults(total_objective=1.5),
        validation=core.ValStepResults(
            head_metrics={
                'head': core.AccumulatedMetrics(mean=0.6)
            }
        ),
    )
    d = results.as_dict

    assert d['training']['total_objective'] == 1.5
    assert d['validation']['head_metrics']['head']['mean'] == 0.6


def test_session_step_results_track_weighted_heads():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.5),
            'head2': core.AccumulatedMetrics(mean=0.9),
        }
    )
    results = core.SessionStepResults(validation=validation)
    results.track('iou', {'head2': 2.0})

    assert results.target_metrics == pytest.approx(0.9)


def test_session_step_results_track_gem():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.5),
            'head2': core.AccumulatedMetrics(mean=0.8),
        },
        mtl_metrics={'gem': 0.94},
    )
    results = core.SessionStepResults(validation=validation)
    results.track('gem', None)

    assert results.target_metrics == pytest.approx(0.94)


def test_session_step_results_track_single_head_forces_iou():
    validation = core.ValStepResults(
        head_metrics={
            'head1': core.AccumulatedMetrics(mean=0.8)
        }
    )
    results = core.SessionStepResults(validation=validation)
    results.track('gem', None)

    assert results.target_objective == 'IoU from head1'
    assert results.target_metrics == pytest.approx(0.8)


# ----- TrainStepResults
def test_train_step_results_clear():
    results = core.TrainStepResults(
        total_objective=3.14,
        head_losses={'focal': 1.2, 'dice': 0.8},
        regularization={'consistency': 0.01},
    )

    assert results.total_objective == 3.14
    assert results.head_losses == {'focal': 1.2, 'dice': 0.8}
    assert results.regularization == {'consistency': 0.01}

    results.clear()

    assert results.total_objective == 0.0
    assert results.head_losses == {'focal': 0.0, 'dice': 0.0}
    assert results.regularization == {'consistency': 0.0}


# ----- ValStepResults & InferStepResults
@pytest.mark.parametrize('cls', [core.ValStepResults, core.InferStepResults])
def test_val_infer_step_results_as_dict_shape(cls):
    results = cls(
        head_metrics={'base_head': core.AccumulatedMetrics(mean=0.5)},
        mtl_metrics={'consistency': 0.2}
    )
    d = results.as_dict

    assert d == {
        'head_metrics': {
            'base_head': {
                'cmatrix': [],
                'mean': 0.5,
                'ious': {},
                'ac_mean': 0.0,
                'ac_ious': {}
            }
        },
        'mtl_metrics': {
            'consistency': 0.2
        }
    }


@pytest.mark.parametrize('cls', [core.ValStepResults, core.InferStepResults])
def test_val_infer_step_results_as_dict_returns_snapshot(cls):
    results = cls(
        head_metrics={'base_head': core.AccumulatedMetrics(mean=0.5)},
        mtl_metrics={'consistency': 0.2}
    )
    d = results.as_dict # snapshot here for later JSON serialization
    results.mtl_metrics['consistency'] = 0.1 # mutated afterward

    assert d['mtl_metrics']['consistency'] == 0.2


# ----- AccumulatedMetrics
def test_accumulated_metrics_as_dict():
    metrics = core.AccumulatedMetrics(
        cmatrix=[[1, 2], [3, 4]],
        mean=0.5,
        ious={'cls_1': 0.3},
        ac_mean=0.6,
        ac_ious={'cls_1': 0.6},
    )
    d = metrics.as_dict

    assert d == {
        'cmatrix': [[1, 2], [3, 4]],
        'mean': 0.5,
        'ious': {'cls_1': 0.3},
        'ac_mean': 0.6,
        'ac_ious': {'cls_1': 0.6},
    }


def test_accumulated_metrics_as_str_list():
    metrics = core.AccumulatedMetrics(
        mean=0.5,
        ious={'1': 0.4, '2': 0.6},
    )

    assert metrics.as_str_list == [
        'Mean IoU (all): 0.5000',
        'Class IoU (all): cls1=0.4000|cls2=0.6000',
    ]


def test_accumulated_metrics_as_str_list_w_active():
    metrics = core.AccumulatedMetrics(
        mean=0.5,
        ious={'1': 0.4},
        ac_mean=0.7,
        ac_ious={'1': 0.7},
    )

    assert metrics.as_str_list == [
        'Mean IoU (all): 0.5000',
        'Class IoU (all): cls1=0.4000',
        'Mean IoU (active): 0.7000',
        'Class IoU (active): cls1=0.7000',
    ]


def test_accumulated_metrics_lock():
    metrics = core.AccumulatedMetrics()
    metrics.lock()

    with pytest.raises(AttributeError):
        metrics.mean = 0.5
    with pytest.raises(AttributeError):
        metrics.ious = {}


# ----- _track_metrics(...)
def test_track_metrics_returns_gem_metric():
    result = session_results._track_metrics(
        {},
        metric_name='gem',
        mtl_metrics={'gem': 0.91},
    )

    assert result == pytest.approx(0.91)


@pytest.mark.parametrize('mtl_metrics', [None, {}, {'iou': 0.5}])
def test_track_metrics_gem_requires_metric(mtl_metrics):
    with pytest.raises(ValueError, match='No valid MTL metrics provided'):
        session_results._track_metrics(
            {},
            metric_name='gem',
            mtl_metrics=mtl_metrics,
        )


def test_track_metrics_iou_uses_requested_heads():
    metrics = {
        'head1': core.AccumulatedMetrics(mean=0.5),
        'head2': core.AccumulatedMetrics(mean=0.8),
    }
    result = session_results._track_metrics(
        metrics,
        metric_name='iou',
        track_heads={'head2': 2.0},
    )

    assert result == pytest.approx(0.8)


def test_track_metrics_iou_defaults_to_first_head():
    metrics = {
        'head1': core.AccumulatedMetrics(mean=0.4),
        'head2': core.AccumulatedMetrics(mean=0.9),
    }
    result = session_results._track_metrics(
        metrics,
        metric_name='iou',
    )

    assert result == pytest.approx(0.4)


@pytest.mark.parametrize('track_heads', [[], 'head1', 1, 3.14])
def test_track_metrics_invalid_tracking_head(track_heads):
    metrics = {'head1': core.AccumulatedMetrics(mean=0.5)}
    with pytest.raises(ValueError, match='Invalid tracking head'):
        session_results._track_metrics(
            metrics,
            metric_name='iou',
            track_heads=track_heads,
        )


def test_track_metrics_invalid_metric_name():
    metrics = {}
    with pytest.raises(ValueError, match='Invalid metric name'):
        session_results._track_metrics(
            metrics,
            metric_name='accuracy',
        )


# ----- _get_mean_iou(...)
def test_get_mean_iou_single_head_uses_mean():
    metrics = {'head1': core.AccumulatedMetrics(mean=0.75)}
    result = session_results._get_mean_iou(metrics, None)

    assert result == pytest.approx(0.75)


def test_get_mean_iou_prefers_active_mean():
    metrics = {'head': core.AccumulatedMetrics(mean=0.60, ac_mean=0.85)}
    result = session_results._get_mean_iou(metrics, None)

    assert result == pytest.approx(0.85)


def test_get_mean_iou_multiple_heads_unweighted():
    metrics = {
        'head1': core.AccumulatedMetrics(mean=0.5),
        'head2': core.AccumulatedMetrics(mean=0.7),
    }
    result = session_results._get_mean_iou(metrics, None)

    assert result == pytest.approx(0.6)


def test_get_mean_iou_weighted():
    metrics = {
        'head1': core.AccumulatedMetrics(mean=0.5),
        'head2': core.AccumulatedMetrics(mean=0.8),
    }
    weights = {'head1': 1.0, 'head2': 3.0}
    result = session_results._get_mean_iou(metrics, weights)
    expected = (1.0 * 0.5 + 3.0 * 0.8) / 4.0

    assert result == pytest.approx(expected)


def test_get_mean_iou_missing_weight_defaults_to_one():
    metrics = {
        'head1': core.AccumulatedMetrics(mean=0.5),
        'head2': core.AccumulatedMetrics(mean=0.9),
    }
    weights = {'head1': 2.0}
    result = session_results._get_mean_iou(metrics, weights)
    expected = (2.0 * 0.5 + 1.0 * 0.9) / 3.0

    assert result == pytest.approx(expected)


def test_get_mean_iou_active_mean_selected_per_head():
    metrics = {
        'head1': core.AccumulatedMetrics(mean=0.4, ac_mean=0.8),
        'head2': core.AccumulatedMetrics(mean=0.6, ac_mean=0.0),
    }
    result = session_results._get_mean_iou(metrics, None)
    expected = (0.8 + 0.6) / 2

    assert result == pytest.approx(expected)
