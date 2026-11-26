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

'''Unit test suite for `landseg.visual.diagnostics` module.'''

# standard imports
import pathlib
# third-party imports
import matplotlib
import numpy
import pytest
import torch
# local imports
import landseg.visual as visual

matplotlib.use('Agg')


# ----- Confusion Matrix Tests
def test_plot_confusion_matrix_tensor(tmp_path: pathlib.Path):
    '''
    Given: A 2D PyTorch confusion matrix tensor.
    When: Calling `plot_confusion_matrix` with `normalize=True`.
    Then: Return a figure and save the PNG to disk.
    '''
    cm_tensor = torch.tensor(
        [[50, 10], [5, 35]],
        dtype=torch.int64,
    )
    save_path = tmp_path / 'cm.png'
    fig = visual.plot_confusion_matrix(
        cm_tensor,
        normalize=True,
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_confusion_matrix_numpy_with_labels(tmp_path: pathlib.Path):
    '''
    Given: A NumPy confusion matrix array and custom class names.
    When: Calling `plot_confusion_matrix` with `normalize=False`.
    Then: Return a figure and save the labeled PNG to disk.
    '''
    cm_arr = numpy.array(
        [[100, 20, 5], [10, 80, 10], [2, 8, 90]],
        dtype=numpy.int32,
    )
    save_path = tmp_path / 'cm_labels.png'
    fig = visual.plot_confusion_matrix(
        cm_arr,
        class_names=['Water', 'Forest', 'Urban'],
        normalize=False,
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_confusion_matrix_invalid_shape():
    '''
    Given: A non-square confusion matrix array.
    When: Calling `plot_confusion_matrix`.
    Then: Raise `ValueError`.
    '''
    cm_invalid = numpy.zeros((3, 4))
    with pytest.raises(ValueError, match='square'):
        visual.plot_confusion_matrix(cm_invalid)


# ----- Per-Class Metrics Tests
def test_plot_per_class_metrics_dict(tmp_path: pathlib.Path):
    '''
    Given: A dictionary mapping class names to metric scores.
    When: Calling `plot_per_class_metrics`.
    Then: Return a figure and save the PNG to disk.
    '''
    metrics = {
        'Water': 0.88,
        'Forest': 0.76,
        'Urban': 0.65,
        'Wetland': 0.81,
    }
    save_path = tmp_path / 'per_class.png'
    fig = visual.plot_per_class_metrics(
        metrics,
        metric_name='IoU',
        save_path=save_path,
        show=False,
    )
    assert fig is not None
    assert save_path.is_file()


def test_plot_per_class_metrics_tensor():
    '''
    Given: A 1D PyTorch tensor of per-class metric scores.
    When: Calling `plot_per_class_metrics` with class names.
    Then: Return a rendered matplotlib figure.
    '''
    tensor_scores = torch.tensor([0.9, 0.75, 0.82], dtype=torch.float32)
    fig = visual.plot_per_class_metrics(
        tensor_scores,
        class_names=['A', 'B', 'C'],
        show=False,
    )
    assert fig is not None


def test_plot_per_class_metrics_invalid_type():
    '''
    Given: An unsupported string metrics payload.
    When: Calling `plot_per_class_metrics`.
    Then: Raise `TypeError`.
    '''
    with pytest.raises(TypeError, match='Unsupported'):
        visual.plot_per_class_metrics('invalid_string_payload')
