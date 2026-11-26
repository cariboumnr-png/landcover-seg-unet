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
import matplotlib.pyplot
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
    '''
    Given: A 2D integer class tensor and color palette dictionary.
    When: Calling `colorize`.
    Then: Return a 3D `uint8` RGB tensor with mapped pixel colors.
    '''
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
    '''
    Given: A 3D tensor instead of a 2D class tensor.
    When: Calling `colorize`.
    Then: Raise `ValueError`.
    '''
    canvas = torch.zeros((2, 2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match='2D'):
        visual.colorize(canvas)


def test_colorize_default_error_palette():
    '''
    Given: A 2D class tensor and `DEFAULT_ERROR_PALETTE`.
    When: Calling `colorize`.
    Then: Return an RGB tensor with canonical error palette colors.
    '''
    canvas = torch.tensor([[0, 1]], dtype=torch.long)
    rgb = visual.colorize(canvas, palette=visual.DEFAULT_ERROR_PALETTE)
    assert rgb.shape == (3, 1, 2)
    # 0 -> [255, 140, 0] (orange)
    assert rgb[:, 0, 0].tolist() == [255, 140, 0]
    # 1 -> [40, 40, 40] (dark gray)
    assert rgb[:, 0, 1].tolist() == [40, 40, 40]


def test_colorize_reference_palettes():
    '''
    Given: A 2D class tensor with `NLCD_16` and `FORESTRY_FRI` palettes.
    When: Calling `colorize`.
    Then: Return RGB tensors matching each reference palette.
    '''
    canvas = torch.tensor([[1, 2], [8, 9]], dtype=torch.long)
    nlcd_rgb = visual.colorize(canvas, palette=visual.NLCD_16_PALETTE)
    assert nlcd_rgb.shape == (3, 2, 2)
    assert nlcd_rgb[:, 0, 0].tolist() == [70, 107, 159]

    fri_rgb = visual.colorize(canvas, palette=visual.FORESTRY_FRI_PALETTE)
    assert fri_rgb.shape == (3, 2, 2)
    assert fri_rgb[:, 0, 0].tolist() == [34, 139, 34]


def test_colorize_continuous_viridis():
    '''
    Given: A 2D float tensor and `viridis` colormap.
    When: Calling `colorize_continuous`.
    Then: Return a non-empty 3D `uint8` RGB tensor.
    '''
    canvas = torch.tensor([[0.0, 0.5], [1.0, 0.2]], dtype=torch.float32)
    rgb = visual.colorize_continuous(canvas, cmap='viridis')
    assert rgb.shape == (3, 2, 2)
    assert rgb.dtype == torch.uint8
    # non-empty, non-zero values generated
    assert rgb.sum() > 0


def test_colorize_continuous_invalid_dim():
    '''
    Given: A 3D float tensor instead of a 2D map.
    When: Calling `colorize_continuous`.
    Then: Raise `ValueError`.
    '''
    canvas = torch.zeros((2, 2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match='2D'):
        visual.colorize_continuous(canvas)


# ----- Curves Plotting Tests
def test_generate_training_curves_multi_head(tmp_path: pathlib.Path):
    '''
    Given: A JSON file with multi-head step metrics and GEM scores.
    When: Calling `generate_training_curves`.
    Then: Save a non-empty PNG and return its path.
    '''
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

    plots_dir = tmp_path / 'plots'
    saved = visual.generate_training_curves(json_path, plots_dir)
    assert saved == plots_dir / 'training_curves.png'
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_generate_training_curves_from_step_summaries(
    tmp_path: pathlib.Path,
):
    '''
    Given: Serialized `SessionStepSummary` records persisted to JSON.
    When: Calling `generate_training_curves`.
    Then: Save a non-empty PNG and return its path.
    '''
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

    saved = visual.generate_training_curves(json_path, tmp_path / 'plots')
    assert saved.is_file()
    assert saved.stat().st_size > 0


def test_generate_training_curves_empty_steps(tmp_path: pathlib.Path):
    '''
    Given: A `step_results.json` file containing an empty list.
    When: Calling `generate_training_curves`.
    Then: Raise `ValueError`.
    '''
    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist([])
    with pytest.raises(ValueError, match='No training step records found'):
        visual.generate_training_curves(json_path, tmp_path / 'plots')


def test_generate_training_curves_nonexistent_path(tmp_path: pathlib.Path):
    '''
    Given: A path to a non-existent `step_results.json` file.
    When: Calling `generate_training_curves`.
    Then: Raise `FileNotFoundError`.
    '''
    missing = tmp_path / 'nonexistent.json'
    with pytest.raises(
        FileNotFoundError, match='Specified path does not exist'
    ):
        visual.generate_training_curves(missing, tmp_path / 'plots')


def test_generate_training_curves_no_validation_metrics(
    tmp_path: pathlib.Path,
):
    '''
    Given: Step records with training loss and no validation metrics.
    When: Calling `generate_training_curves`.
    Then: Save the loss curve PNG cleanly and return its path.
    '''
    step_data = [
        {'epoch_in_phase': 1, 'training': {'total_objective': 1.5}},
        {'epoch_in_phase': 2, 'training': {'total_objective': 1.1}},
    ]
    json_path = tmp_path / 'step_results.json'
    artifacts.Controller[list](json_path).persist(step_data)

    saved = visual.generate_training_curves(json_path, tmp_path / 'plots')
    assert saved.is_file()


# ----- Qualitative Inspection Tests
def test_plot_qualitative_predictions(tmp_path: pathlib.Path):
    '''
    Given: Synthetic image, ground truth, prediction, and error arrays.
    When: Calling `plot_qualitative_predictions`.
    Then: Return a figure and save the 4-panel PNG to disk.
    '''
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
def test_generate_prediction_previews_skips_empty(tmp_path: pathlib.Path):
    '''
    Given: `None` or incomplete `InferStepResults` payloads.
    When: Calling `generate_prediction_previews`.
    Then: Skip rendering, write no PNGs, and return an empty dictionary.
    '''
    specs = _make_mock_dataspecs()
    previews_dir = tmp_path / 'previews'

    assert not visual.generate_prediction_previews(
        None,
        specs,
        previews_dir,
    )
    assert not visual.generate_prediction_previews(
        core.InferStepResults(),
        specs,
        previews_dir,
    )
    assert not visual.generate_prediction_previews(
        core.InferStepResults(infer_inputs=torch.zeros((4, 64, 64))),
        specs,
        previews_dir,
    )
    assert not list(tmp_path.glob('**/*.png'))


def test_generate_training_curves_from_file_and_dir(tmp_path: pathlib.Path):
    '''
    Given: A `step_results.json` file path and its parent directory.
    When: Calling `generate_training_curves` with both paths.
    Then: Load records and save a PNG in both cases.
    '''
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

    # from file path
    saved1 = visual.generate_training_curves(json_path, tmp_path / 'p1')
    assert saved1.is_file()

    # from directory path
    saved2 = visual.generate_training_curves(tmp_path, tmp_path / 'p2')
    assert saved2.is_file()


def test_generate_prediction_previews_renders_pngs(tmp_path: pathlib.Path):
    '''
    Given: Complete `InferStepResults` with labels, preds, and errors.
    When: Calling `generate_prediction_previews`.
    Then: Save per-head preview PNGs and return their paths.
    '''
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
    )
    assert figs['h1'] == previews_dir / 'h1_preview.png'
    assert figs['h1'].is_file()


def test_generate_prediction_previews_unlabeled(tmp_path: pathlib.Path):
    '''
    Given: `InferStepResults` with predictions but no labels or errors.
    When: Calling `generate_prediction_previews`.
    Then: Save per-head preview PNGs cleanly and return their paths.
    '''
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
    )
    assert figs['h1'] == previews_dir / 'h1_preview.png'
    assert figs['h1'].is_file()


# ----- Render Saved Figures Display
def test_render_prediction_preview_loads_existing_png(tmp_path: pathlib.Path):
    '''
    Given: A run directory with `step_results.json` and a preview PNG.
    When: Calling `render_prediction_preview` on the results root.
    Then: Display the PNG without error and leave no figures open.
    '''
    run_dir = _make_run_dir(tmp_path)
    _save_sample_png(run_dir / 'previews' / 'head1_preview.png')

    visual.render_prediction_preview(tmp_path)
    assert not matplotlib.pyplot.get_fignums()


def test_render_prediction_preview_missing_pngs(tmp_path: pathlib.Path):
    '''
    Given: A run directory with an empty `previews` directory.
    When: Calling `render_prediction_preview`.
    Then: Raise `FileNotFoundError`.
    '''
    run_dir = _make_run_dir(tmp_path)
    (run_dir / 'previews').mkdir()
    with pytest.raises(FileNotFoundError, match='No preview PNGs'):
        visual.render_prediction_preview(tmp_path)


def test_render_training_curves_loads_existing_png(tmp_path: pathlib.Path):
    '''
    Given: A run directory with `step_results.json` and a curves PNG.
    When: Calling `render_training_curves` on the results root.
    Then: Display the PNG without error and leave no figures open.
    '''
    run_dir = _make_run_dir(tmp_path)
    _save_sample_png(run_dir / 'plots' / 'training_curves.png')

    visual.render_training_curves(tmp_path)
    assert not matplotlib.pyplot.get_fignums()


def test_render_training_curves_missing_results(tmp_path: pathlib.Path):
    '''
    Given: A results directory without any `step_results.json`.
    When: Calling `render_training_curves`.
    Then: Raise `FileNotFoundError`.
    '''
    with pytest.raises(FileNotFoundError, match='No step_results.json'):
        visual.render_training_curves(tmp_path)


# ----- private helpers
def _make_run_dir(root: pathlib.Path) -> pathlib.Path:
    '''Create a run directory marked by an empty `step_results.json`.'''
    run_dir = root / 'run_0001'
    run_dir.mkdir(parents=True)
    artifacts.Controller[list](run_dir / 'step_results.json').persist([])
    return run_dir


def _save_sample_png(fpath: pathlib.Path) -> None:
    '''Save a small line-plot PNG to the given path.'''
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = matplotlib.pyplot.subplots()
    ax.plot([0, 1], [0, 1])
    fig.savefig(fpath)
    matplotlib.pyplot.close(fig)


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
