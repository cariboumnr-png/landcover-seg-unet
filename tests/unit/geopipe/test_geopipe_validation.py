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

'''Unit tests for `landseg.geopipe.validation` contract facade.'''

# standard imports
import types
import typing
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe as geopipe
import landseg.geopipe.core as geo_core
import landseg.geopipe.validation as validation


# ----- grid & manifest facade tests
def test_validate_grid_specs_extracts_dict_details(monkeypatch):
    '''
    Given: A stage `grid.validators.grid_specs` return value with or
        without `as_dict()`.
    When: Calling `geopipe.validate_grid_specs`.
    Then: Return normalized dict details or `{}` when unavailable.
    '''
    dummy_cfg = typing.cast(typing.Any, object())

    monkeypatch.setattr(
        'landseg.geopipe.grid.validators.grid_specs',
        lambda _cfg: types.SimpleNamespace(
            as_dict=lambda: {'crs': 'EPSG:3161', 'tile_size': (256, 256)}
        ),
    )
    assert geopipe.validate_grid_specs(dummy_cfg) == {
        'crs': 'EPSG:3161',
        'tile_size': (256, 256),
    }

    monkeypatch.setattr(
        'landseg.geopipe.grid.validators.grid_specs',
        lambda _cfg: types.SimpleNamespace(grid_id='grid_no_as_dict'),
    )
    assert not geopipe.validate_grid_specs(dummy_cfg)


def test_validate_grid_reference_returns_summary(monkeypatch, tmp_path):
    '''
    Given: A valid grid reference validator result.
    When: Calling `geopipe.validate_grid_reference`.
    Then: Return `grid_id`, `tiles`, `grid_fpath`, and `report_fpath`.
    '''
    report_fp = tmp_path / 'report.json'
    grid_fp = tmp_path / 'grid.json'
    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        lambda _fp: ({(0, 0): 'w0', (256, 0): 'w1'}, 'gid_01', grid_fp),
    )

    summary = geopipe.validate_grid_reference(report_fp)
    assert summary == {
        'grid_id': 'gid_01',
        'tiles': 2,
        'grid_fpath': str(grid_fp),
        'report_fpath': str(report_fp),
    }


def test_validate_manifest_contract_returns_count(monkeypatch, tmp_path):
    '''
    Given: A manifest contract validator returning two entries.
    When: Calling `geopipe.validate_manifest_contract`.
    Then: Return `entry_count` and `dataset_manifest`.
    '''
    manifest_fp = tmp_path / 'manifest.json'
    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.manifest_contract',
        lambda _fp: ({'name': 'img'}, {'name': 'lbl'}),
    )

    summary = geopipe.validate_manifest_contract(manifest_fp)
    assert summary == {
        'entry_count': 2,
        'dataset_manifest': str(manifest_fp),
    }


# ----- pool compatibility tests
def test_validate_pool_compatibility_success_and_errors(monkeypatch, tmp_path):
    '''
    Given: Schema and world grid report paths.
    When: Calling `geopipe.validate_pool_compatibility`.
    Then: Return compatibility metadata or propagate wrapped grid or
        pool errors.
    '''
    schema_fp = tmp_path / 'schema.json'
    report_fp = tmp_path / 'report.json'

    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        lambda _fp: (types.SimpleNamespace(gid='gid_utm17'), 'gid_utm17', _fp),
    )
    monkeypatch.setattr(
        'landseg.geopipe.ingest.validators.pool_compatibility',
        lambda _s, _g: None,
    )

    before_schema = geopipe.validate_pool_compatibility(schema_fp, report_fp)
    assert before_schema == {
        'grid_id': 'gid_utm17',
        'schema_fpath': str(schema_fp),
        'schema_exists': False,
    }

    schema_fp.write_text('{}', encoding='utf-8')
    after_schema = geopipe.validate_pool_compatibility(schema_fp, report_fp)
    assert after_schema['schema_exists'] is True

    # grid reference failure wraps cleanly
    def _bad_grid(_fp):
        raise RuntimeError('corrupt grid')

    monkeypatch.setattr(
        'landseg.geopipe.harmonize.validators.grid_reference',
        _bad_grid,
    )
    with pytest.raises(ValueError, match='World grid reference invalid: corrupt grid'):
        validation.validate_pool_compatibility(schema_fp, report_fp)


# ----- ingestion contract validation tests
def test_validate_feature_specs_skip_missing_and_pass(monkeypatch, tmp_path):
    '''
    Given: Harmonization and ingestion path containers.
    When: Pending batch is absent, lacks feature rasters, or validates
        cleanly.
    Then: Return skip outcome, raise `ValueError`, or return validated
        spec fields.
    '''
    harm_paths = artifacts.HarmonizationPaths(tmp_path / 'harm')
    ing_paths = artifacts.IngestionPaths(tmp_path / 'ingest')

    # 1. no pending batches -> skip
    monkeypatch.setattr(
        'landseg.geopipe.ingest.resolve_pending_ingestion_batches',
        lambda *_args, **_kw: [],
    )
    skip_res = geopipe.validate_feature_specs(harm_paths, ing_paths)
    assert skip_res['status'] == 'skip'
    assert 'unable to resolve harmonization batch' in skip_res['message']

    # 2. missing feature raster -> ValueError
    monkeypatch.setattr(
        'landseg.geopipe.ingest.resolve_pending_ingestion_batches',
        lambda *_args, **_kw: [{'run_uid': 'r1', 'run_id': 'run_0001'}],
    )
    monkeypatch.setattr(
        'landseg.geopipe.ingest.build_ingestion_context',
        lambda *_args, **_kw: types.SimpleNamespace(
            features=None,
            labels=None,
            grid=types.SimpleNamespace(gid='gid_1'),
        ),
    )
    with pytest.raises(ValueError, match='No harmonized feature raster'):
        geopipe.validate_feature_specs(harm_paths, ing_paths)

    # 3. valid context -> calls stage feature_specs validator
    recorded: dict[str, typing.Any] = {}
    img_fp = tmp_path / 'features.vrt'
    lbl_fp = tmp_path / 'labels.vrt'
    monkeypatch.setattr(
        'landseg.geopipe.ingest.build_ingestion_context',
        lambda *_args, **_kw: types.SimpleNamespace(
            features=img_fp,
            labels=lbl_fp,
            grid=types.SimpleNamespace(gid='gid_1'),
        ),
    )
    monkeypatch.setattr(
        'landseg.geopipe.ingest.validators.feature_specs',
        lambda feat, lbl, *, add_spectral=None, add_topo=None: recorded.update({
            'feat': feat,
            'lbl': lbl,
            'add_spectral': add_spectral,
            'add_topo': add_topo,
        }),
    )

    pass_res = geopipe.validate_feature_specs(
        harm_paths,
        ing_paths,
        add_spectral=['ndvi'],
        add_topo=['slope'],
    )
    assert pass_res == {
        'status': 'pass',
        'add_spectral': ['ndvi'],
        'add_topo': ['slope'],
    }
    assert recorded == {
        'feat': img_fp,
        'lbl': lbl_fp,
        'add_spectral': ['ndvi'],
        'add_topo': ['slope'],
    }


def test_validate_raster_alignment_skip_and_pass(monkeypatch, tmp_path):
    '''
    Given: Harmonization and ingestion path containers.
    When: Calling `geopipe.validate_raster_alignment`.
    Then: Return skip when batch unavailable or pass with raster and
        grid details.
    '''
    harm_paths = artifacts.HarmonizationPaths(tmp_path / 'harm')
    ing_paths = artifacts.IngestionPaths(tmp_path / 'ingest')

    def _raise_missing(*_args, **_kw):
        raise artifacts.ArtifactError('manifest missing')

    monkeypatch.setattr(
        'landseg.geopipe.ingest.resolve_pending_ingestion_batches',
        _raise_missing,
    )
    skip_res = geopipe.validate_raster_alignment(harm_paths, ing_paths)
    assert skip_res['status'] == 'skip'
    assert 'manifest missing' in skip_res['message']

    img_fp = tmp_path / 'features.vrt'
    mock_grid = types.SimpleNamespace(gid='gid_aligned')
    monkeypatch.setattr(
        'landseg.geopipe.ingest.resolve_pending_ingestion_batches',
        lambda *_args, **_kw: [{'run_uid': 'r1', 'run_id': 'run_0001'}],
    )
    monkeypatch.setattr(
        'landseg.geopipe.ingest.build_ingestion_context',
        lambda *_args, **_kw: types.SimpleNamespace(
            features=img_fp,
            labels=None,
            grid=mock_grid,
        ),
    )
    called: list[tuple[typing.Any, ...]] = []
    monkeypatch.setattr(
        'landseg.geopipe.ingest.validators.raster_alignment',
        lambda feat, grid, lbl: called.append((feat, grid, lbl)),
    )

    pass_res = geopipe.validate_raster_alignment(harm_paths, ing_paths)
    assert pass_res == {
        'status': 'pass',
        'features': str(img_fp),
        'labels': None,
        'grid_id': 'gid_aligned',
    }
    assert called == [(img_fp, mock_grid, None)]


# ----- prepare semantics test
def test_validate_prepare_semantics_fetches_data_schema(monkeypatch, tmp_path):
    '''
    Given: An ingestion schema file path and dataset view config.
    When: Calling `geopipe.validate_prepare_semantics`.
    Then: Fetch via `DataSchema` schema ID and return plain head and
        feature lists.
    '''
    schema_fp = tmp_path / 'schema.json'
    captured_schema_ids: list[str | None] = []

    def _fake_fetch(self):
        captured_schema_ids.append(self.schema_id)
        return {'schema_id': geo_core.data_schema.SCHEMA_ID}

    monkeypatch.setattr(artifacts.Controller, 'fetch', _fake_fetch)
    monkeypatch.setattr(
        'landseg.geopipe.prepare.validators.dataset_semantics',
        lambda _schm, _cfg=None: (
            types.SimpleNamespace(names=('red', 'nir')),
            types.SimpleNamespace(head_names=['landcover', 'landcover_group']),
            'landcover_group',
        ),
    )

    summary = geopipe.validate_prepare_semantics(schema_fp)
    assert captured_schema_ids == [geo_core.data_schema.SCHEMA_ID]
    assert summary == {
        'focal_head': 'landcover_group',
        'features': ['red', 'nir'],
        'target_heads': ['landcover', 'landcover_group'],
        'schema_fpath': str(schema_fp),
    }
