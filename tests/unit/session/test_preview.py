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

'''Unit tests for `landseg.session` preview inference utilities.'''

# standard imports
import dataclasses
import pathlib
# third-party imports
import numpy
import pytest
import torch
# local imports
import landseg.core as core
import landseg.session as session


# ----- `select_representative_block` tests
def test_select_representative_block_prefers_class_diversity(
    tmp_path: pathlib.Path,
):
    '''
    Given: Blocks where the first block has highest class diversity.
    When: Calling `session.select_representative_block`.
    Then: Deterministically select the most diverse block (not middle).
    '''
    b0 = tmp_path / 'row_0000_col_0000.npz'
    lbl0 = numpy.zeros((2, 64, 64), dtype=numpy.uint8)
    lbl0[0, 10:30, 10:30] = 1
    lbl0[0, 30:50, 30:50] = 2
    numpy.savez(b0, label=lbl0)

    b1 = tmp_path / 'row_0000_col_0064.npz'
    numpy.savez(b1, label=numpy.zeros((2, 64, 64), dtype=numpy.uint8))

    b2 = tmp_path / 'row_0000_col_0128.npz'
    numpy.savez(b2, label=numpy.ones((2, 64, 64), dtype=numpy.uint8))

    blocks = {
        'row_0000_col_0000': b0,
        'row_0000_col_0064': b1,
        'row_0000_col_0128': b2,
    }
    name, path = session.select_representative_block(blocks, label_key='label')
    assert name == 'row_0000_col_0000'
    assert path == b0


def test_select_representative_block_single_and_empty(
    tmp_path: pathlib.Path,
):
    '''
    Given: Empty block dict or single-entry block dict.
    When: Calling `session.select_representative_block`.
    Then: Raise `ValueError` if empty; return sole entry if single.
    '''
    with pytest.raises(ValueError, match='No test blocks'):
        session.select_representative_block({})

    only = tmp_path / 'only.npz'
    name, path = session.select_representative_block({'only': only})
    assert name == 'only'
    assert path == only


def test_select_representative_block_fallback_on_missing_key(
    tmp_path: pathlib.Path,
):
    '''
    Given: Blocks lacking the requested label key.
    When: Calling `session.select_representative_block`.
    Then: Fall back deterministically to the middle sorted entry.
    '''
    paths = {}
    for idx in range(3):
        fpath = tmp_path / f'b{idx}.npz'
        numpy.savez(fpath, image=numpy.zeros((3, 8, 8), dtype=numpy.float32))
        paths[f'b{idx}'] = fpath

    name, path = session.select_representative_block(paths, label_key='label')
    assert name == 'b1'
    assert path == paths['b1']


# ----- `run_preview_inference` tests
def test_run_preview_inference_skips_missing_inputs(
    tmp_path: pathlib.Path,
    dataspecs,
    mock_model,
    session_config,
):
    '''
    Given: Empty test split or non-existent checkpoint file.
    When: Calling `session.run_preview_inference`.
    Then: Return None without raising.
    '''
    no_test_specs = dataclasses.replace(
        dataspecs,
        splits=dataclasses.replace(dataspecs.splits, test={}),
    )
    missing_ckpt = tmp_path / 'missing.pt'
    assert session.run_preview_inference(
        checkpoint_fpath=missing_ckpt,
        dataspecs=no_test_specs,
        model=mock_model,
        config=session_config,
        device='cpu',
    ) is None

    assert session.run_preview_inference(
        checkpoint_fpath=missing_ckpt,
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        device='cpu',
    ) is None


def test_run_preview_inference_contiguous_mosaic(
    tmp_path: pathlib.Path,
    dataspecs,
    mock_model,
    session_config,
):
    '''
    Given: Contiguous test block grid and saved model checkpoint.
    When: Calling `session.run_preview_inference`.
    Then: Load checkpoint and return stitched `InferStepResults`.
    '''
    ckpt_path = _save_model_checkpoint(tmp_path / 'best.pt', mock_model)

    results = session.run_preview_inference(
        checkpoint_fpath=ckpt_path,
        dataspecs=dataspecs,
        model=mock_model,
        config=session_config,
        device='cpu',
    )

    assert isinstance(results, core.InferStepResults)
    assert isinstance(results.infer_inputs, torch.Tensor)
    assert results.infer_inputs.shape == (4, 256, 256)
    assert 'head_1' in results.infer_preds
    assert results.infer_preds['head_1'].shape == (256, 256)
    assert results.infer_labels['head_1'].shape == (256, 256)
    assert results.infer_errors['head_1'].shape == (256, 256)


def test_run_preview_inference_non_contiguous_fallback_and_gating(
    tmp_path: pathlib.Path,
    dataspecs,
    session_config,
):
    '''
    Given: Non-contiguous split `(0, 0)` and parent head gating.
    When: Calling `session.run_preview_inference`.
    Then: Pick representative block and gate inactive child outputs.
    '''
    b0 = tmp_path / 'b0.npz'
    img0 = numpy.zeros((4, 256, 256), dtype=numpy.float32)
    lbl0 = numpy.ones((2, 256, 256), dtype=numpy.uint8)
    lbl0[0, :64, :64] = 2
    numpy.savez(b0, image=img0, label=lbl0)

    b1 = tmp_path / 'b1.npz'
    numpy.savez(
        b1,
        image=numpy.zeros((4, 256, 256), dtype=numpy.float32),
        label=numpy.zeros((2, 256, 256), dtype=numpy.uint8),
    )

    gated_specs = dataclasses.replace(
        dataspecs,
        meta=dataclasses.replace(dataspecs.meta, test_blks_grid=(0, 0)),
        heads=dataclasses.replace(
            dataspecs.heads,
            head_parent={'head_1': None, 'head_2': 'head_1'},
            head_parent_cls={'head_1': None, 'head_2': 2},
        ),
        splits=dataclasses.replace(
            dataspecs.splits,
            test={'b0': b0, 'b1': b1},
        ),
        domains=dataclasses.replace(
            dataspecs.domains,
            test={'ids_domain': None, 'vec_domain': None},
        ),
    )

    model = _FixedHeadPredictionModel()
    ckpt_path = _save_model_checkpoint(tmp_path / 'gated_best.pt', model)

    results = session.run_preview_inference(
        checkpoint_fpath=ckpt_path,
        dataspecs=gated_specs,
        model=model,
        config=session_config,
        device='cpu',
    )

    assert isinstance(results, core.InferStepResults)
    assert isinstance(results.infer_inputs, torch.Tensor)
    assert results.infer_inputs.shape == (4, 256, 256)
    # head_1 predicts class 1 everywhere, so child head_2 (parent_cls=2) is gated to 0
    assert torch.all(results.infer_preds['head_1'] == 1)
    assert torch.all(results.infer_preds['head_2'] == 0)


def test_run_preview_inference_unlabeled_block(
    tmp_path: pathlib.Path,
    dataspecs,
    mock_model,
    session_config,
):
    '''
    Given: Unlabeled test block with placeholder label array.
    When: Calling `session.run_preview_inference`.
    Then: Return stitched inputs and preds with empty labels dict.
    '''
    blk = tmp_path / 'unlabeled.npz'
    numpy.savez(
        blk,
        image=numpy.zeros((4, 256, 256), dtype=numpy.float32),
        label=numpy.array([1]),
    )
    unlabeled_specs = dataclasses.replace(
        dataspecs,
        meta=dataclasses.replace(dataspecs.meta, test_blks_grid=(1, 1)),
        splits=dataclasses.replace(dataspecs.splits, test={'unlabeled': blk}),
        domains=dataclasses.replace(
            dataspecs.domains,
            test={'ids_domain': None, 'vec_domain': None},
        ),
    )
    ckpt_path = _save_model_checkpoint(tmp_path / 'unlabeled.pt', mock_model)

    results = session.run_preview_inference(
        checkpoint_fpath=ckpt_path,
        dataspecs=unlabeled_specs,
        model=mock_model,
        config=session_config,
        device='cpu',
    )

    assert isinstance(results, core.InferStepResults)
    assert isinstance(results.infer_inputs, torch.Tensor)
    assert results.infer_inputs.shape == (4, 256, 256)
    assert 'head_1' in results.infer_preds
    assert results.infer_labels == {}
    assert results.infer_errors == {}


# ----- private helpers
def _save_model_checkpoint(
    fpath: pathlib.Path,
    model: torch.nn.Module,
) -> pathlib.Path:
    '''Save minimal checkpoint matching `load_checkpoint` format.'''
    torch.save(
        {'model': model.state_dict(), 'metric': 0.5, 'epoch': 1, 'step': 1},
        fpath,
    )
    return fpath


class _FixedHeadPredictionModel(torch.nn.Module):
    '''Deterministic model emitting fixed class logits per head.'''

    def __init__(self) -> None:
        super().__init__()
        self.spatial_divisor = 16
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.active_heads: list[str] | None = None
        self.logit_adjust_alpha: float = 1.0

    def set_active_heads(self, active_heads: list[str] | None) -> None:
        '''Set active heads.'''
        self.active_heads = active_heads

    def set_frozen_heads(self, frozen_heads: list[str] | None) -> None:
        '''Ignore frozen heads.'''
        _ = frozen_heads

    def reset_heads(self) -> None:
        '''Reset active heads.'''
        self.active_heads = None

    def set_logit_adjust_alpha(self, alpha: float) -> None:
        '''Record logit adjust alpha.'''
        self.logit_adjust_alpha = alpha

    def forward(self, x, ids_domain=None, vec_domain=None):
        '''Emit fixed logits for `head_1` and `head_2`.'''
        _ = ids_domain, vec_domain
        bsz, _, h, w = x.shape
        h1 = torch.zeros((bsz, 2, h, w), dtype=torch.float32)
        h1[:, 0] = 5.0
        h2 = torch.zeros((bsz, 3, h, w), dtype=torch.float32)
        h2[:, 1] = 5.0
        return {'head_1': h1, 'head_2': h2}
