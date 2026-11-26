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

'''Unit tests for diagnostic overfit `DataSpecs` builder (`geopipe`).'''
# pylint: disable=protected-access

# standard imports
import pathlib
import types
import typing
# third-party imports
import numpy
import pytest
import torch
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe as geopipe
import landseg.geopipe.core as geo_core
import landseg.geopipe.diagnostic as diagnostic


# ----- helpers
def _make_block(
    *,
    name: str = 'row_0000_col_0000',
    image: numpy.ndarray | None = None,
    label_count: dict[str, list[int]] | None = None,
    valid_ratio: float = 1.0,
) -> geo_core.DataBlock:
    '''Create a minimal in-memory `DataBlock` for diagnostic tests.'''
    img = (
        image
        if image is not None
        else numpy.arange(2 * 4 * 4, dtype=numpy.float32).reshape(2, 4, 4)
    )
    lbl = numpy.ones((1, 4, 4), dtype=numpy.uint8)
    mask = numpy.ones((4, 4), dtype=bool)
    counts = (
        label_count
        if label_count is not None
        else {'landcover': [10, 6], 'original': [16]}
    )
    manifest: typing.Any = {
        'block_name': name,
        'has_label': True,
        'image_band_map': {'blue': 0, 'green': 1},
        'image_nodata': -9999.0,
        'label_band_map': {'landcover': 0},
        'label_nodata': 255,
        'label_ignore_index': 255,
        'label_ignore_cls': {'landcover': [255]},
        'label_num_cls': {'landcover': 2},
        'label_cls_names': {'landcover': ['conifer', 'deciduous']},
        'label_cls_clr_map': {},
        'label_taxonomy': {'landcover': {'profile': 'demo_profile'}},
        'valid_ratios': {'image': valid_ratio},
        'image_stats': {},
        'label_count': counts,
        'label_entropy': {},
    }
    return geo_core.DataBlock(
        data=geo_core.DataBlockArrays(image=img, label=lbl, valid_mask=mask),
        manifest=manifest,
    )


# ----- existing cached block tests
def test_build_overfit_dataspec_reuses_existing_block_and_knowledge(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: A cached diagnostic `.npz` block and an existing knowledge
        folder.
    When: Calling `geopipe.build_overfit_dataspec`.
    Then: Reuse cached block, build neutral class counts, and resolve
        knowledge.
    '''
    save_dir = tmp_path / 'overfit_test'
    save_dir.mkdir(parents=True)
    block_fp = save_dir / 'test_row_0000_col_0000.npz'
    _make_block().save(block_fp)

    kroot = tmp_path / 'knowledge'
    kroot.mkdir()
    kp = artifacts.KnowledgePaths(root=kroot)
    artifacts.Controller[torch.Tensor](
        kp.similarity_matrix_fpath('demo_profile')
    ).persist(torch.eye(2))
    monkeypatch.setattr(
        'landseg.knowledge.validate_specs',
        lambda profile, mapping, num_cls, knowledge_root: {
            'profile': profile,
            'verified_classes': num_cls,
        },
    )

    specs = geopipe.build_overfit_dataspec(
        save_dpath=save_dir,
        harmonization_paths=artifacts.HarmonizationPaths(tmp_path / 'harm'),
        ingestion_paths=artifacts.IngestionPaths(tmp_path / 'ing'),
        knowledge_root=kroot,
    )

    assert specs.name == 'overfit_single_block'
    assert specs.meta.image_specs.height_width == 4
    assert specs.meta.image_specs.band_map == {'blue': 0, 'green': 1}
    assert specs.meta.label_specs.ignore_index == 255
    assert specs.heads.class_counts == {'landcover': [1, 1]}
    assert specs.heads.logits_adjust == {'landcover': [1.0, 1.0]}
    assert specs.heads.head_parent == {'landcover': None}
    assert specs.heads.taxonomy == {
        'landcover': {'profile': 'demo_profile', 'verified_classes': 2}
    }
    assert torch.equal(
        specs.heads.similarity_matrices['landcover'], torch.eye(2)
    )
    assert specs.splits.train == {'row_0000_col_0000': block_fp.resolve()}
    assert specs.splits.val == {'row_0000_col_0000': block_fp.resolve()}
    assert not specs.splits.test


# ----- block construction fallback tests
def test_build_overfit_dataspec_creates_and_normalizes_block(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Empty overfit directory and candidate rasters lacking one
        class under strict all-class filtering.
    When: Calling `geopipe.build_overfit_dataspec`.
    Then: Fall back to relaxed filtering, normalize image values, and
        save `.npz`.
    '''
    save_dir = tmp_path / 'overfit_new'
    harm_paths = artifacts.HarmonizationPaths(tmp_path / 'harm')
    ing_paths = artifacts.IngestionPaths(tmp_path / 'ing')

    monkeypatch.setattr(
        diagnostic.ingest,
        'resolve_pending_ingestion_batches',
        lambda *_a, **_kw: [{'run_uid': 'u1', 'run_id': 'run_0001'}],
    )
    monkeypatch.setattr(
        diagnostic.ingest,
        'build_ingestion_context',
        lambda *_a, **_kw: types.SimpleNamespace(
            has_data=True,
            grid=object(),
            features=tmp_path / 'feat.vrt',
            labels=tmp_path / 'lbl.vrt',
        ),
    )
    monkeypatch.setattr(
        diagnostic.mapper_mod,
        'map_rasters_to_grid',
        lambda *_a, **_kw: types.SimpleNamespace(
            image={(0, 0): 'win_0'},
            label={(0, 0): 'lbl_0'},
        ),
    )
    monkeypatch.setattr(
        diagnostic.assembler.RasterReadContext,
        'from_rasters',
        lambda *_a, **_kw: types.SimpleNamespace(
            label_specs={'landcover': object()}
        ),
    )
    monkeypatch.setattr(
        diagnostic.assembler,
        'read_block_raster_data',
        lambda *_a, **_kw: object(),
    )
    monkeypatch.setattr(
        diagnostic.assembler,
        'build_data_block',
        lambda name, _data: _make_block(
            name=name,
            label_count={'landcover': [16, 0]},
            valid_ratio=1.0,
        ),
    )

    specs = geopipe.build_overfit_dataspec(
        save_dpath=save_dir,
        harmonization_paths=harm_paths,
        ingestion_paths=ing_paths,
    )

    saved_files = list(save_dir.glob('test_*.npz'))
    assert len(saved_files) == 1
    assert saved_files[0].resolve() in specs.splits.train.values()
    loaded = geo_core.DataBlock.load(saved_files[0])
    assert numpy.isclose(numpy.mean(loaded.data.image), 0.0, atol=1e-5)
    assert numpy.isclose(numpy.std(loaded.data.image), 1.0, atol=1e-5)


def test_build_overfit_dataspec_error_cases(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    '''
    Given: Missing pending harmonization batches or missing label specs.
    When: Invoking diagnostic target head / dataspec builders.
    Then: Raise explicit ValueError errors.
    '''
    with pytest.raises(ValueError, match='No label specifications found'):
        diagnostic._resolve_target_head({}, None)

    assert (
        diagnostic._resolve_target_head(
            {'a': typing.cast(typing.Any, object()), 'b': typing.cast(typing.Any, object())},
            ['b'],
        )
        == 'b'
    )

    monkeypatch.setattr(
        diagnostic.ingest,
        'resolve_pending_ingestion_batches',
        lambda *_a, **_kw: [],
    )
    with pytest.raises(ValueError, match='No pending harmonization runs'):
        geopipe.build_overfit_dataspec(
            save_dpath=tmp_path / 'empty',
            harmonization_paths=artifacts.HarmonizationPaths(tmp_path / 'harm'),
            ingestion_paths=artifacts.IngestionPaths(tmp_path / 'ing'),
        )
