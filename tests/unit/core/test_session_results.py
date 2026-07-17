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

# pylint: disable=protected-access

'''Unit tests for dataclasses and functions in session_results.py.'''

# third-party imports
import pytest
# local imports
import landseg.core as core
import landseg.core.session_results as session_results


# ----- `SessionStepSummary`
def test_session_step_summary_as_dict_contains_expected_keys():
    '''
    Given: A SessionStepSummary instance.
    When: Accessing the as_dict property.
    Then: Return a dictionary with the expected keys.
    '''
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
    '''
    Given: A SessionStepSummary initialized with nested results.
    When: Accessing the as_dict property.
    Then: Correctly serialize the nested step result dicts.
    '''
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
    '''
    Given: A SessionStepSummary with empty raw metrics.
    When: Accessing the as_dict property.
    Then: Return None for missing step results fields.
    '''
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
    '''
    Given: A SessionStepSummary instance.
    When: Accessing the as_dict property and mutating the source fields.
    Then: The dict return value is unaffected by subsequent mutations.
    '''
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


# ----- `SessionStepResults`
def test_session_step_results_target_objective_default():
    '''
    Given: SessionStepResults with validation metrics.
    When: Fetching target_objective.
    Then: Return default validation metric name.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(head={'mean': 0.5})
    )
    results = core.SessionStepResults(validation=validation)

    assert results.target_objective == 'IoU from head'


def test_session_step_results_target_objective_weighted():
    '''
    Given: SessionStepResults tracked with weighted heads.
    When: Fetching target_objective.
    Then: Return the weighted combination metric name.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(head1={}, head2={})
    )
    results = core.SessionStepResults(validation=validation)
    results.track('iou', {'head1': 0.25, 'head2': 0.75})

    assert results.target_objective == 'IoU = head1*0.25 + head2*0.75'


def test_session_step_results_target_objective_gem():
    '''
    Given: SessionStepResults tracked with gem metric.
    When: Fetching target_objective.
    Then: Return MTL Global Exact Match objective description.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(head1={}, head2={}),
        mtl_metrics={'gem': 0.9},
    )
    results = core.SessionStepResults(validation=validation)
    results.track('gem', None)

    assert results.target_objective == 'Global Exact Match over [head1 & head2]'


def test_session_step_results_target_objective_invalid_metric():
    '''
    Given: SessionStepResults tracked with invalid metric name.
    When: Fetching target_objective.
    Then: Raise a ValueError.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(head1={}, head2={})
    )
    results = core.SessionStepResults(validation=validation)
    object.__setattr__(results, '_metric_name', 'foo')

    with pytest.raises(ValueError, match='Invalid metric name'):
        _ = results.target_objective


def test_session_step_results_target_metrics_no_validation():
    '''
    Given: SessionStepResults lacking validation data.
    When: Fetching target_metrics.
    Then: Return negative infinity.
    '''
    results = core.SessionStepResults()

    assert results.target_metrics == -float('inf')


def test_session_step_results_target_metrics_default_first_head():
    '''
    Given: SessionStepResults with unweighted multi-head validation.
    When: Fetching target_metrics.
    Then: Return target metrics for the first head.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(
            head1={'mean': 0.6},
            head2={'mean': 0.9}
        )
    )
    results = core.SessionStepResults(validation=validation)

    assert results.target_metrics == pytest.approx(0.6)


def test_session_step_results_inference_metrics_no_inference():
    '''
    Given: SessionStepResults lacking inference results.
    When: Fetching inference_metrics.
    Then: Return negative infinity.
    '''
    results = core.SessionStepResults()

    assert results.inference_metrics == -float('inf')


def test_session_step_results_inference_metrics_default_first_head():
    '''
    Given: SessionStepResults with unweighted multi-head inference.
    When: Fetching inference_metrics.
    Then: Return target metrics for the first head.
    '''
    inference = core.InferStepResults(
        head_metrics=_make_head_metrics(
            head1={'mean': 0.75},
            head2={'mean': 0.9}
        )
    )
    results = core.SessionStepResults(inference=inference)

    assert results.inference_metrics == pytest.approx(0.75)


def test_session_step_results_as_dict():
    '''
    Given: SessionStepResults containing training and validation steps.
    When: Accessing the as_dict property.
    Then: Return compiled step dictionary.
    '''
    results = core.SessionStepResults(
        training=core.TrainStepResults(total_objective=1.5),
        validation=core.ValStepResults(
            head_metrics=_make_head_metrics(head={'mean': 0.6})
        ),
    )
    d = results.as_dict

    assert d['training']['total_objective'] == 1.5
    assert d['validation']['head_metrics']['head']['mean'] == 0.6


def test_session_step_results_track_weighted_heads():
    '''
    Given: SessionStepResults containing multi-head validation data.
    When: Tracking a weighted metric selection.
    Then: Evaluate the target metric value correctly.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(
            head1={'mean': 0.6},
            head2={'mean': 0.9}
        )
    )
    results = core.SessionStepResults(validation=validation)
    results.track('iou', {'head2': 2.0})

    assert results.target_metrics == pytest.approx(0.9)


def test_session_step_results_track_gem():
    '''
    Given: SessionStepResults containing MTL validation metrics.
    When: Tracking MTL gem metric.
    Then: Evaluate the target metric value correctly.
    '''
    validation = core.ValStepResults(
        mtl_metrics={'gem': 0.94},
    )
    results = core.SessionStepResults(validation=validation)
    results.track('gem', None)

    assert results.target_metrics == pytest.approx(0.94)


def test_session_step_results_track_single_head_forces_iou():
    '''
    Given: SessionStepResults with a single head validation block.
    When: Requesting a gem metric tracker.
    Then: Force target_objective back to single-head IoU format.
    '''
    validation = core.ValStepResults(
        head_metrics=_make_head_metrics(head={'mean': 0.8})
    )
    results = core.SessionStepResults(validation=validation)
    results.track('gem', None)

    assert results.target_objective == 'IoU from head'
    assert results.target_metrics == pytest.approx(0.8)


# ----- `TrainStepResults`
def test_train_step_results_clear():
    '''
    Given: A populated TrainStepResults instance.
    When: Clearing the results.
    Then: Reset all losses and regularizations to baseline.
    '''
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


# ----- `ValStepResults` & `InferStepResults`
@pytest.mark.parametrize('cls', [core.ValStepResults, core.InferStepResults])
def test_val_infer_step_results_as_dict_shape(cls):
    '''
    Given: A ValStepResults or InferStepResults instance.
    When: Calling as_dict.
    Then: Return a dictionary with structured metrics lists.
    '''
    results = cls(
        head_metrics=_make_head_metrics(base_head={'mean': 0.5}),
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
    '''
    Given: A ValStepResults or InferStepResults instance.
    When: Accessing the as_dict property and mutating later metrics.
    Then: Retain the original snapshot values in the dictionary.
    '''
    results = cls(
        mtl_metrics={'consistency': 0.2}
    )
    d = results.as_dict
    results.mtl_metrics['consistency'] = 0.1

    assert d['mtl_metrics']['consistency'] == 0.2


# ----- `AccumulatedMetrics`
def test_accumulated_metrics_as_dict():
    '''
    Given: An AccumulatedMetrics instance.
    When: Calling as_dict.
    Then: Return a serialized dictionary of values.
    '''
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
    '''
    Given: An AccumulatedMetrics instance without active class splits.
    When: Formatting metrics string lists.
    Then: Return formatted string representations for all classes.
    '''
    metrics = core.AccumulatedMetrics(
        mean=0.5,
        ious={'1': 0.4, '2': 0.6},
    )

    assert metrics.as_str_list == [
        'Mean IoU (all): 0.5000',
        'Class IoU (all): cls1=0.4000|cls2=0.6000',
    ]


def test_accumulated_metrics_as_str_list_w_active():
    '''
    Given: An AccumulatedMetrics instance with active class splits.
    When: Formatting metrics string lists.
    Then: Include formatted string representations for active classes.
    '''
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
    '''
    Given: An AccumulatedMetrics instance.
    When: Locking the metrics.
    Then: Freeze internal properties and raise an AttributeError on edit.
    '''
    metrics = core.AccumulatedMetrics()
    metrics.lock()

    with pytest.raises(AttributeError):
        metrics.mean = 0.5
    with pytest.raises(AttributeError):
        metrics.ious = {}


# ----- `_track_metrics`(...)
def test_track_metrics_returns_gem_metric():
    '''
    Given: MTL metrics mapping containing gem.
    When: Invoking _track_metrics.
    Then: Return the gem metric score.
    '''
    result = session_results._track_metrics(
        {},
        metric_name='gem',
        mtl_metrics={'gem': 0.91},
    )

    assert result == pytest.approx(0.91)


@pytest.mark.parametrize('mtl_metrics', [None, {}, {'iou': 0.5}])
def test_track_metrics_gem_requires_metric(mtl_metrics):
    '''
    Given: Missing or empty MTL metrics.
    When: Requesting tracking for the gem metric.
    Then: Raise a ValueError.
    '''
    with pytest.raises(ValueError, match='No valid MTL metrics provided'):
        session_results._track_metrics(
            {},
            metric_name='gem',
            mtl_metrics=mtl_metrics,
        )


def test_track_metrics_iou_uses_requested_heads():
    '''
    Given: Multi-head validation metrics.
    When: Requesting tracking on specific head names.
    Then: Extract score for the target heads.
    '''
    metrics = _make_head_metrics(
        head1={'mean': 0.5},
        head2={'mean': 0.8}
    )
    result = session_results._track_metrics(
        metrics,
        metric_name='iou',
        track_heads={'head2': 2.0},
    )

    assert result == pytest.approx(0.8)


def test_track_metrics_iou_defaults_to_first_head():
    '''
    Given: Multi-head validation metrics.
    When: Running tracking with no specified heads map.
    Then: Default to the first head in the metrics catalog.
    '''
    metrics = _make_head_metrics(
        head1={'mean': 0.4},
        head2={'mean': 0.8}
    )
    result = session_results._track_metrics(
        metrics,
        metric_name='iou',
    )

    assert result == pytest.approx(0.4)


@pytest.mark.parametrize('track_heads', [[], 'head', 1, 3.14])
def test_track_metrics_invalid_tracking_head(track_heads):
    '''
    Given: Invalid formatting for target track heads.
    When: Running tracking.
    Then: Raise a ValueError.
    '''
    metrics = _make_head_metrics(head={'mean': 0.75})
    with pytest.raises(ValueError, match='Invalid tracking head'):
        session_results._track_metrics(
            metrics,
            metric_name='iou',
            track_heads=track_heads,
        )


def test_track_metrics_invalid_metric_name():
    '''
    Given: An invalid metric objective name.
    When: Running tracking.
    Then: Raise a ValueError.
    '''
    metrics = {}
    with pytest.raises(ValueError, match='Invalid metric name'):
        session_results._track_metrics(
            metrics,
            metric_name='accuracy',
        )


# ----- `_get_mean_iou`(...)
def test_get_mean_iou_single_head_uses_mean():
    '''
    Given: Metrics containing a single validation head.
    When: Running _get_mean_iou.
    Then: Return the head mean value.
    '''
    metrics = _make_head_metrics(head={'mean': 0.75})
    result = session_results._get_mean_iou(metrics, None)

    assert result == pytest.approx(0.75)


def test_get_mean_iou_prefers_active_mean():
    '''
    Given: Metrics containing active class validation splits.
    When: Running _get_mean_iou.
    Then: Prefer returning the active class mean over default mean.
    '''
    metrics = _make_head_metrics(head={'mean': 0.6, 'ac_mean': 0.85})
    result = session_results._get_mean_iou(metrics, None)

    assert result == pytest.approx(0.85)


def test_get_mean_iou_multiple_heads_unweighted():
    '''
    Given: Multiple validation heads without specified weights.
    When: Running _get_mean_iou.
    Then: Return the flat average of the head scores.
    '''
    metrics = _make_head_metrics(
        head1={'mean': 0.5},
        head2={'mean': 0.7}
    )
    result = session_results._get_mean_iou(metrics, None)

    assert result == pytest.approx(0.6)


def test_get_mean_iou_weighted():
    '''
    Given: Multiple validation heads with weights.
    When: Running _get_mean_iou.
    Then: Return the weighted average score.
    '''
    metrics = _make_head_metrics(
        head1={'mean': 0.5},
        head2={'mean': 0.8}
    )
    weights = {'head1': 1.0, 'head2': 3.0}
    result = session_results._get_mean_iou(metrics, weights)
    expected = (1.0 * 0.5 + 3.0 * 0.8) / 4.0

    assert result == pytest.approx(expected)


def test_get_mean_iou_missing_weight_defaults_to_one():
    '''
    Given: Multiple validation heads with partial weights.
    When: Running _get_mean_iou.
    Then: Default missing weights to a factor of 1.0.
    '''
    metrics = _make_head_metrics(
        head1={'mean': 0.5},
        head2={'mean': 0.9}
    )
    weights = {'head1': 2.0}
    result = session_results._get_mean_iou(metrics, weights)
    expected = (2.0 * 0.5 + 1.0 * 0.9) / 3.0

    assert result == pytest.approx(expected)


def test_get_mean_iou_active_mean_selected_per_head():
    '''
    Given: Multi-head validation where only some heads have active means.
    When: Running _get_mean_iou.
    Then: Resolve active/default means properly per head.
    '''
    metrics = _make_head_metrics(
        head1={'mean': 0.4, 'ac_mean': 0.8},
        head2={'mean': 0.6, 'ac_mean': 0.0}
    )
    result = session_results._get_mean_iou(metrics, None)
    expected = (0.8 + 0.6) / 2

    assert result == pytest.approx(expected)


# ----- builders
def _make_head_metrics(**heads) -> dict[str, core.AccumulatedMetrics]:
    '''Construct a head_metrics dictionary.'''
    return {
        name: core.AccumulatedMetrics(**kwargs)
        for name, kwargs in heads.items()
    }
