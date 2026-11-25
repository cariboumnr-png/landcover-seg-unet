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

'''Unit test suite for `landseg.visual` pure visualization module.'''

# standard imports
import pathlib
# third-party imports
import matplotlib
import numpy
import pytest
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.core as core
import landseg.visual as visual

matplotlib.use('Agg')


# ----- Colorization Tests
def test_colorize_tensor():
    '''Test mapping 2D class integer tensor to 3D RGB tensor.'''
    canvas = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)
    palette = {
        '0': [0, 0, 0],
        '1': [255, 0, 0],
        '2': [0, 255, 0],
    }
    rgb = visual.colorize(canvas, palette=palette)
    assert rgb.shape == (3, 2, 2)
    assert rgb.dtype == torch.uint8
    # pixel (0, 1) has class 1 -> red [255, 0, 0]
    assert rgb[:, 0, 1].tolist() == [255, 0, 0]


def test_colorize_invalid_dim():
    '''Test that non-2D tensor raises ValueError.'''
    canvas = torch.zeros((2, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match='2D'):
        visual.colorize(canvas)


def test_colorize_default_error_palette():
    '''Test colorizing with canonical DEFAULT_ERROR_PALETTE.'''
    canvas = torch.tensor([[0, 1]], dtype=torch.long)
    rgb = visual.colorize(canvas, palette=visual.DEFAULT_ERROR_PALETTE)
    assert rgb.shape == (3, 1, 2)
    # 0 -> [255, 140, 0] (orange)
    assert rgb[:, 0, 0].tolist() == [255, 140, 0]
    # 1 -> [40, 40, 40] (dark gray)
    assert rgb[:, 0, 1].tolist() == [40, 40, 40]


def test_colorize_reference_palettes():
    '''Test colorizing with NLCD and FRI forestry palettes.'''
    canvas = torch.tensor([[1, 2], [8, 9]], dtype=torch.long)
    nlcd_rgb = visual.colorize(canvas, palette=visual.NLCD_16_PALETTE)
    assert nlcd_rgb.shape == (3, 2, 2)
    assert nlcd_rgb[:, 0, 0].tolist() == [70, 107, 159]

    fri_rgb = visual.colorize(canvas, palette=visual.FORESTRY_FRI_PALETTE)
    assert fri_rgb.shape == (3, 2, 2)
    assert fri_rgb[:, 0, 0].tolist() == [34, 139, 34]


def test_colorize_continuous_viridis():
    '''Test continuous float tensor colorization.'''
    canvas = torch.tensor([[0.0, 0.5], [1.0, 0.2]], dtype=torch.float32)
    rgb = visual.colorize_continuous(canvas, cmap='viridis')
    assert rgb.shape == (3, 2, 2)
    assert rgb.dtype == torch.uint8
    # non-empty, non-zero values generated
    assert rgb.sum() > 0


def test_colorize_continuous_invalid_dim():
    '''Test continuous colorization rejects non-2D tensors.'''
    canvas = torch.zeros((2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match='2D'):
        visual.colorize_continuous(canvas)


# ----- Curves Plotting Tests
def test_plot_training_curves_multi_head(tmp_path: pathlib.Path):
    '''Test plotting curves with multiple active heads.'''
    steps = [
        {
            'epoch_in_phase': 1,
            'training': {'total_objective': 1.5},
            'validation': {
                'head_metrics': {
                    'landcover': {'mean': 0.40},
                    'species': {'mean': 0.30},
                },
                'mtl_metrics': {'gem': 0.35},
            },
        },
        {
            'epoch_in_phase': 2,
            'training': {'total_objective': 1.0},
            'validation': {
                'head_metrics': {
                    'landcover': {'mean': 0.65},
                    'species': {'mean': 0.55},
                },
                'mtl_metrics': {'gem': 0.60},
            },
        },
    ]

    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist(steps)

    out_file = tmp_path / 'curves.png'
    fig = visual.plot_training_curves(json_path, save_path=out_file, show=False)
    assert fig is not None
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_training_curves_from_step_summaries(tmp_path: pathlib.Path):
    '''Test plotting curves from serialized SessionStepSummary records.'''
    training1 = core.TrainStepResults(total_objective=1.5)
    validation1 = core.ValStepResults(
        head_metrics={'landcover': core.AccumulatedMetrics(mean=0.45)}
    )
    step1 = core.SessionStepSummary(
        phase_name='train',
        phase_index=0,
        phase_max_epoch=10,
        epoch_in_phase=1,
        global_epoch=1,
        is_phase_end=False,
        is_run_end=False,
        stop_reason=None,
        val_metrics_name='IoU from landcover',
        val_metrics_value=0.45,
        best_value_so_far=0.45,
        best_epoch_so_far=1,
        is_best_epoch=True,
        raw_metrics=core.SessionStepResults(
            training=training1,
            validation=validation1,
        ),
    )

    training2 = core.TrainStepResults(total_objective=1.0)
    validation2 = core.ValStepResults(
        head_metrics={'landcover': core.AccumulatedMetrics(mean=0.60)}
    )
    step2 = core.SessionStepSummary(
        phase_name='train',
        phase_index=0,
        phase_max_epoch=10,
        epoch_in_phase=2,
        global_epoch=2,
        is_phase_end=False,
        is_run_end=False,
        stop_reason=None,
        val_metrics_name='IoU from landcover',
        val_metrics_value=0.60,
        best_value_so_far=0.60,
        best_epoch_so_far=2,
        is_best_epoch=True,
        raw_metrics=core.SessionStepResults(
            training=training2,
            validation=validation2,
        ),
    )

    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist(
        [step1.as_dict, step2.as_dict]
    )

    out_file = tmp_path / 'summary_curves.png'
    fig = visual.plot_training_curves(json_path, save_path=out_file, show=False)
    assert fig is not None
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_training_curves_empty_steps(tmp_path: pathlib.Path):
    '''Test that empty step results file raises ValueError.'''
    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist([])
    with pytest.raises(ValueError, match='No training step records found'):
        visual.plot_training_curves(json_path)


def test_plot_training_curves_nonexistent_path(tmp_path: pathlib.Path):
    '''Test that non-existent path raises FileNotFoundError.'''
    missing = tmp_path / 'nonexistent.json'
    with pytest.raises(
        FileNotFoundError, match='Specified path does not exist'
    ):
        visual.plot_training_curves(missing)


def test_plot_training_curves_no_validation_metrics(tmp_path: pathlib.Path):
    '''Test plotting curves when no validation head metrics are recorded.'''
    step_data = [
        {'epoch_in_phase': 1, 'training': {'total_objective': 1.5}},
        {'epoch_in_phase': 2, 'training': {'total_objective': 1.1}},
    ]
    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist(step_data)

    fig = visual.plot_training_curves(json_path, show=False)
    assert fig is not None


# ----- Qualitative Inspection Tests
def test_plot_qualitative_predictions(tmp_path: pathlib.Path):
    '''Test 3-panel and 4-panel qualitative segmentation plots.'''
    img = numpy.random.rand(64, 64, 3).astype(numpy.float32)
    gt = numpy.random.randint(0, 3, size=(64, 64), dtype=numpy.int32)
    pred = numpy.random.randint(0, 3, size=(64, 64), dtype=numpy.int32)
    err = (pred != gt).astype(numpy.int32)

    # 4-panel with error overlay
    out_file = tmp_path / 'preview_4p.png'
    fig = visual.plot_qualitative_predictions(
        img, gt, pred, err,
        save_path=out_file,
        show=False,
    )
    assert fig is not None
    assert out_file.is_file()


# ----- Preview Generation Empty / Rendering Tests
def test_generate_prediction_previews_skips_empty_results():
    '''Test skipping preview generation when inference inputs/preds are absent.'''
    specs = _make_mock_dataspecs()

    assert not visual.generate_prediction_previews(None, specs)
    assert not visual.generate_prediction_previews(
        core.InferStepResults(),
        specs,
    )
    assert not visual.generate_prediction_previews(
        core.InferStepResults(infer_inputs=torch.zeros((4, 64, 64))),
        specs,
    )


def test_plot_training_curves_from_file_and_dir(tmp_path: pathlib.Path):
    '''Test loading step_results from json file and directory.'''
    step_data = [
        {
            'epoch_in_phase': 1,
            'training': {'total_objective': 2.0},
            'validation': {
                'head_metrics': {'h1': {'mean': 0.45}},
                'mtl_metrics': {'gem': 0.40},
            },
        }
    ]
    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist(step_data)

    # From file path
    fig1 = visual.plot_training_curves(json_path, show=False)
    assert fig1 is not None

    # From directory path
    fig2 = visual.plot_training_curves(tmp_path, show=False)
    assert fig2 is not None


def test_generate_prediction_previews_renders_pngs(tmp_path: pathlib.Path):
    '''Test pure preview rendering from InferStepResults and DataSpecs.'''
    specs = _make_mock_dataspecs(
        band_map={'red': 2, 'green': 1, 'blue': 0, 'nir': 3},
        color_map={'0': [0, 0, 0], '1': [255, 0, 0]},
    )
    infer_inputs = torch.zeros((4, 64, 64), dtype=torch.float32)
    infer_inputs[2] = 1.0  # red band per band_map
    infer_labels = {'h1': torch.ones((64, 64), dtype=torch.long)}
    infer_preds = {'h1': torch.ones((64, 64), dtype=torch.long)}
    infer_errors = {'h1': torch.ones((64, 64), dtype=torch.long)}

    infer_results = core.InferStepResults(
        infer_inputs=infer_inputs,
        infer_labels=infer_labels,
        infer_preds=infer_preds,
        infer_errors=infer_errors,
    )

    previews_dir = tmp_path / 'previews'
    figs = visual.generate_prediction_previews(
        infer_results=infer_results,
        dataspecs=specs,
        previews_dpath=previews_dir,
        save_pngs=True,
        show=False,
    )
    assert 'h1' in figs
    assert (previews_dir / 'h1_preview.png').is_file()


def test_generate_prediction_previews_unlabeled(tmp_path: pathlib.Path):
    '''Test pure preview rendering when labels and errors are absent.'''
    specs = _make_mock_dataspecs()
    infer_results = core.InferStepResults(
        infer_inputs=torch.zeros((1, 64, 64), dtype=torch.float32),
        infer_preds={'h1': torch.ones((64, 64), dtype=torch.long)},
    )

    previews_dir = tmp_path / 'previews'
    figs = visual.generate_prediction_previews(
        infer_results=infer_results,
        dataspecs=specs,
        previews_dpath=previews_dir,
        save_pngs=True,
        show=False,
    )
    assert 'h1' in figs
    assert (previews_dir / 'h1_preview.png').is_file()


# ----- Render Prediction Preview Display
def test_render_prediction_preview_loads_existing_png(tmp_path: pathlib.Path):
    '''Test render_prediction_preview finds and displays existing PNGs.'''
    prev_dir = tmp_path / 'previews'
    prev_dir.mkdir(parents=True)
    sample_png = prev_dir / 'head1_preview.png'

    # Save a small image
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(sample_png)
    matplotlib.pyplot.close(fig)

    rendered_fig = visual.render_prediction_preview(tmp_path, show=False)
    assert rendered_fig is not None


# ----- private helpers
def _make_mock_dataspecs(
    *,
    band_map: dict[str, int] | None = None,
    color_map: dict[str, list[int]] | None = None,
) -> core.DataSpecs:
    '''Create minimal `DataSpecs` for preview rendering tests.'''
    return core.DataSpecs(
        name='mock',
        mode='test_only',
        meta=core.Meta(
            test_blks_grid=(1, 1),
            blk_bytes=100,
            label_color_map=color_map or {},
            image_specs=core.Meta.Image(
                height_width=64,
                array_key='image',
                band_map=band_map or {},
            ),
            label_specs=core.Meta.Label(
                array_key='label',
                ignore_index=0,
            ),
        ),
        heads=core.Heads(
            class_counts={'h1': [2]},
            logits_adjust={'h1': [0.0]},
            head_parent={'h1': None},
            head_parent_cls={'h1': None},
            taxonomy={},
            similarity_matrices={},
        ),
        splits=core.Splits(train={}, val={}, test={}),
        domains=core.Domains(
            train=core.Domains.Dom(ids_domain=None, vec_domain=None),
            val=core.Domains.Dom(ids_domain=None, vec_domain=None),
            test=core.Domains.Dom(ids_domain=None, vec_domain=None),
            ids_num=0,
            vec_dim=0,
        ),
    )
