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

'''Unit test suite for `landseg.notebooks.telemetry` interactive module.'''

# standard imports
import io
import json
import pathlib
import time
# third-party imports
import numpy
import pytest
# local imports
import landseg.notebooks as nb


def test_throttled_progress_timing():
    '''
    Given: A ThrottledProgress instance with 0.1s interval.
    When: Emitting messages rapidly.
    Then: Emits only when threshold time has elapsed.
    '''
    throttler = nb.ThrottledProgress(interval_sec=0.1)
    buf = io.StringIO()

    # First emission should succeed
    assert throttler.emit('First update', stream=buf)
    # Immediate second emission should be throttled
    assert not throttler.emit('Second update', stream=buf)

    # After waiting beyond interval, next emission should succeed
    time.sleep(0.12)
    assert throttler.emit('Third update', stream=buf)


def test_safe_training_session_normal_exit(tmp_path: pathlib.Path):
    '''
    Given: An uninterrupted training block wrapped in safe_training_session.
    When: Block executes to completion.
    Then: State records uninterrupted execution and no emergency checkpoint is written.
    '''
    exp_root = tmp_path / 'exp'
    exp_root.mkdir()

    with nb.safe_training_session(exp_root) as state:
        assert state['interrupted'] is False
        _ = 1 + 1

    assert state['interrupted'] is False
    assert state['emergency_checkpoint'] is None


def test_safe_training_session_handles_keyboard_interrupt(tmp_path: pathlib.Path):
    '''
    Given: A simulated KeyboardInterrupt inside safe_training_session.
    When: Cell is interrupted.
    Then: Checkpoint is preserved and exception is handled gracefully without crash.
    '''
    exp_root = tmp_path / 'exp'
    ckpts_dir = exp_root / 'artifacts' / 'checkpoints'
    ckpts_dir.mkdir(parents=True)

    # Create dummy existing checkpoint
    existing_ckpt = ckpts_dir / 'epoch_0001_last.pt'
    existing_ckpt.write_bytes(b'dummy_weights')

    with nb.safe_training_session(exp_root, reraise=False) as state:
        raise KeyboardInterrupt('Simulated notebook stop button click')

    assert state['interrupted'] is True
    assert state['emergency_checkpoint'] is not None

    emergency_file = ckpts_dir / 'checkpoint_interrupted.pt'
    assert emergency_file.is_file()
    assert emergency_file.read_bytes() == b'dummy_weights'


def test_safe_training_session_reraises_when_requested(tmp_path: pathlib.Path):
    '''
    Given: safe_training_session with reraise=True.
    When: KeyboardInterrupt occurs.
    Then: Saves emergency state and re-raises KeyboardInterrupt.
    '''
    exp_root = tmp_path / 'exp'

    with pytest.raises(KeyboardInterrupt):
        with nb.safe_training_session(exp_root, reraise=True):
            raise KeyboardInterrupt('User stop')


def test_plot_training_curves_from_list(tmp_path: pathlib.Path):
    '''
    Given: List of session step dictionaries.
    When: Calling plot_training_curves.
    Then: Successfully generates and saves the figure.
    '''
    steps = [
        {
            'epoch_in_phase': 1,
            'training': {'total_objective': 0.85},
            'validation': {'mtl_metrics': {'total_loss': 0.78}},
            'validation_metrics_name': 'mean_iou',
            'validation_metrics_value': 0.52,
        },
        {
            'epoch_in_phase': 2,
            'training': {'total_objective': 0.65},
            'validation': {'mtl_metrics': {'total_loss': 0.58}},
            'validation_metrics_name': 'mean_iou',
            'validation_metrics_value': 0.68,
        },
    ]

    out_png = tmp_path / 'curves.png'
    fig = nb.plot_training_curves(steps, save_path=out_png, show=False)
    assert fig is not None
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_plot_training_curves_from_file(tmp_path: pathlib.Path):
    '''
    Given: A step_results.json file on disk.
    When: Passing the directory or file path to plot_training_curves.
    Then: Loads records and builds figure.
    '''
    steps = [
        {
            'epoch_in_phase': 1,
            'training': {'total_objective': 0.9},
            'validation_metrics_name': 'mean_iou',
            'validation_metrics_value': 0.60,
        }
    ]
    json_path = tmp_path / 'step_results.json'
    json_path.write_text(json.dumps(steps), encoding='utf-8')

    fig = nb.plot_training_curves(tmp_path, show=False)
    assert fig is not None


def test_plot_qualitative_predictions(tmp_path: pathlib.Path):
    '''
    Given: Synthetic RGB image, ground truth mask, and prediction mask.
    When: Calling plot_qualitative_predictions.
    Then: Successfully constructs and saves side-by-side plot.
    '''
    img = numpy.random.rand(64, 64, 3).astype(numpy.float32)
    gt = numpy.random.randint(0, 4, size=(64, 64), dtype=numpy.int32)
    pred = numpy.random.randint(0, 4, size=(64, 64), dtype=numpy.int32)

    out_png = tmp_path / 'qualitative.png'
    fig = nb.plot_qualitative_predictions(
        img,
        gt,
        pred,
        save_path=out_png,
        show=False,
    )
    assert fig is not None
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


def test_render_prediction_preview(tmp_path: pathlib.Path):
    '''
    Given: An experiment root workspace directory.
    When: Calling render_prediction_preview.
    Then: Generates and persists the preview comparison image.
    '''
    preview_file = tmp_path / 'artifacts' / 'previews' / 'preview.png'
    fig = nb.render_prediction_preview(tmp_path, save_path=preview_file, show=False)
    assert fig is not None
    assert preview_file.is_file()
