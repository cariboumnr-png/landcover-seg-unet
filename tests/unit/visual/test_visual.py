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
import json
import pathlib
# third-party imports
import matplotlib
import numpy
import pytest
import torch
# local imports
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
                'mtl_metrics': {'total_loss': 1.2},
                'head_metrics': {
                    'landcover': {'mean': 0.40},
                    'species': {'mean': 0.30},
                },
            },
        },
        {
            'epoch_in_phase': 2,
            'training': {'total_objective': 1.0},
            'validation': {
                'mtl_metrics': {'total_loss': 0.8},
                'head_metrics': {
                    'landcover': {'mean': 0.65},
                    'species': {'mean': 0.55},
                },
            },
        },
    ]

    out_file = tmp_path / 'curves.png'
    fig = visual.plot_training_curves(steps, save_path=out_file, show=False)
    assert fig is not None
    assert out_file.is_file()
    assert out_file.stat().st_size > 0


def test_plot_training_curves_empty_steps():
    '''Test that empty steps raise ValueError.'''
    with pytest.raises(ValueError, match='No training step records found'):
        visual.plot_training_curves([])


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


# ----- Representative Block Selection Tests
def test_select_representative_block(tmp_path: pathlib.Path):
    '''Test deterministic selection of representative block with high class diversity.'''
    # Block 1: homogeneous background (only class 0)
    b1_path = tmp_path / 'row_0000_col_0000.npz'
    numpy.savez(b1_path, label=numpy.zeros((2, 64, 64), dtype=numpy.uint8))

    # Block 2: diverse classes (classes 0, 1, 2)
    b2_path = tmp_path / 'row_0000_col_0064.npz'
    lbl2 = numpy.zeros((2, 64, 64), dtype=numpy.uint8)
    lbl2[0, 10:30, 10:30] = 1
    lbl2[0, 30:50, 30:50] = 2
    numpy.savez(b2_path, label=lbl2)

    blocks = {
        'row_0000_col_0000': b1_path,
        'row_0000_col_0064': b2_path,
    }

    name, path = visual.select_representative_block(blocks, label_key='label')
    assert name == 'row_0000_col_0064'
    assert path == b2_path


# ----- Preview Generation Empty / Missing Handling
def test_generate_prediction_previews_skips_when_no_test():
    '''Test skipping preview generation when test split is empty.'''
    specs = core.DataSpecs(
        name='mock',
        mode='default',
        meta=core.Meta(
            test_blks_grid=(0, 0),
            blk_bytes=100,
            label_color_map={},
            image_specs=core.Meta.Image(height_width=256, array_key='image', band_map={}),
            label_specs=core.Meta.Label(array_key='label', ignore_index=[0]),
        ),
        heads=core.Heads(
            class_counts={'h1': [10]},
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

    result = visual.generate_prediction_previews(
        checkpoint_fpath='dummy.pt',
        dataspecs=specs,
    )
    assert result == {}


def test_plot_training_curves_from_file_and_dir(tmp_path: pathlib.Path):
    '''Test loading step_results from json file and directory.'''
    step_data = [
        {
            'epoch_in_phase': 1,
            'training': {'total_objective': 2.0},
            'validation': {
                'mtl_metrics': {'total_loss': 1.8},
                'head_metrics': {'h1': {'mean': 0.45}},
            },
        }
    ]
    json_path = tmp_path / 'step_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(step_data, f)

    # From file path
    fig1 = visual.plot_training_curves(json_path, show=False)
    assert fig1 is not None

    # From directory path
    fig2 = visual.plot_training_curves(tmp_path, show=False)
    assert fig2 is not None


def test_generate_prediction_previews_execution(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    '''Test complete generate_prediction_previews flow with mock model.'''
    blk_path = tmp_path / 'row_0000_col_0000.npz'
    img = numpy.zeros((3, 64, 64), dtype=numpy.float32)
    lbl = numpy.ones((1, 64, 64), dtype=numpy.uint8)
    numpy.savez(blk_path, image=img, label=lbl)

    ckpt_path = tmp_path / 'best.pt'
    ckpt_path.write_bytes(b'dummy_checkpoint')

    specs = core.DataSpecs(
        name='mock',
        mode='default',
        meta=core.Meta(
            test_blks_grid=(0, 0),
            blk_bytes=100,
            label_color_map={'1': [255, 0, 0]},
            image_specs=core.Meta.Image(height_width=64, array_key='image', band_map={'red': 0, 'green': 1, 'blue': 2}),
            label_specs=core.Meta.Label(array_key='label', ignore_index=[0]),
        ),
        heads=core.Heads(
            class_counts={'h1': [2]},
            logits_adjust={'h1': [0.0]},
            head_parent={'h1': None},
            head_parent_cls={'h1': None},
            taxonomy={},
            similarity_matrices={},
        ),
        splits=core.Splits(train={}, val={}, test={'row_0000_col_0000': blk_path}),
        domains=core.Domains(
            train=core.Domains.Dom(ids_domain=None, vec_domain=None),
            val=core.Domains.Dom(ids_domain=None, vec_domain=None),
            test=core.Domains.Dom(ids_domain=None, vec_domain=None),
            ids_num=0,
            vec_dim=0,
        ),
    )

    class MockModel(torch.nn.Module):
        def forward(self, x, **kwargs):
            return {'h1': torch.zeros((x.shape[0], 2, x.shape[2], x.shape[3]))}

    monkeypatch.setattr('landseg.models.build_multihead_unet', lambda **kwargs: MockModel())
    monkeypatch.setattr('landseg.artifacts.load_checkpoint', lambda **kwargs: None)

    previews_dir = tmp_path / 'previews'
    figs = visual.generate_prediction_previews(
        checkpoint_fpath=ckpt_path,
        dataspecs=specs,
        previews_dpath=previews_dir,
        patch_size=64,
        batch_size=1,
        device='cpu',
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
