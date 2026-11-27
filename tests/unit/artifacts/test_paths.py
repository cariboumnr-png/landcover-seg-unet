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

'''Unit tests for `landseg.artifacts.paths`.'''

# standard imports
import os
import pathlib
import time
# third-party imports
import pytest
# local imports
import landseg.artifacts.controller as ctrl_mod
import landseg.artifacts.paths as paths_mod
import landseg.configs as configs_mod


# ----- `ArtifactPaths` & Foundation tests
def test_artifact_paths_hierarchy():
    '''
    Given: A shared root directory string.
    When: Instantiating `ArtifactPaths`.
    Then: Derive legacy stage namespaces beneath the shared root.
    '''
    r = pathlib.Path('/tmp') / 'exp'
    art = paths_mod.ArtifactPaths(root=r)
    assert art.data_harmonization.root == r / 'artifacts' / 'harmonized_data'
    assert art.data_ingestion.root == r / 'artifacts' / 'ingested_data'
    assert art.data_preparation.root == r / 'artifacts' / 'prepared_data'
    assert art.session.root == r / 'results'


def test_artifact_paths_require_explicit_root():
    '''
    Given: An `ArtifactPaths` with no shared root.
    When: Accessing a stage namespace.
    Then: Raise an actionable error instead of guessing a parent.
    '''
    art = paths_mod.ArtifactPaths()
    try:
        _ = art.data_preparation
    except ValueError as e:
        assert 'data.preparation.output_dpath' in str(e)
    else:
        raise AssertionError('Expected ValueError for missing stage root')


def test_artifact_paths_autonomous_stage_roots():
    '''
    Given: Explicit stage roots with no shared experiment root.
    When: Accessing namespaces.
    Then: Resolve each stage independently.
    '''
    art = paths_mod.ArtifactPaths(
        preparation_root=pathlib.Path('/volumes/prepared'),
        session_root=pathlib.Path('/volumes/run_42'),
    )
    assert art.data_preparation.root == pathlib.Path('/volumes/prepared')
    assert art.session.root == pathlib.Path('/volumes/run_42')
    assert art.root is None


def test_artifact_paths_custom_overrides():
    '''
    Given: Explicit per-pipeline sub-root path overrides.
    When: Instantiating `ArtifactPaths` with sub-root parameters.
    Then: Prioritize custom sub-root paths over `exp_root` defaults.
    '''
    art = paths_mod.ArtifactPaths(
        root=pathlib.Path('/tmp/exp'),
        harmonization_root=pathlib.Path('/custom/harmonized'),
        ingestion_root=pathlib.Path('/custom/ingested'),
        preparation_root=pathlib.Path('/custom/prepared'),
        session_root=pathlib.Path('/custom/session'),
    )
    assert art.data_harmonization.root == pathlib.Path('/custom/harmonized')
    assert art.data_ingestion.root == pathlib.Path('/custom/ingested')
    assert art.data_preparation.root == pathlib.Path('/custom/prepared')
    assert art.session.root == pathlib.Path('/custom/session')


def test_artifact_paths_from_config():
    '''
    Given: A `RootConfig` instance with explicit output directories.
    When: Instantiating `ArtifactPaths` via `from_config`.
    Then: Correctly extract root and stage output directory paths.
    '''
    cfg = configs_mod.RootConfig()
    cfg.execution.exp_root = '/tmp/exp'
    cfg.data.harmonization.output_dpath = '/tmp/exp/artifacts/harmonized_data'
    cfg.data.ingestion.output_dpath = '/tmp/exp/artifacts/ingested_data'
    cfg.data.preparation.output_dpath = '/tmp/exp/artifacts/prepared_data'
    cfg.session.output_dpath = '/tmp/exp/results'
    art = paths_mod.ArtifactPaths.from_config(cfg)
    assert art.root == pathlib.Path('/tmp/exp')
    assert art.data_harmonization.root == pathlib.Path(
        '/tmp/exp/artifacts/harmonized_data'
    )
    assert art.data_ingestion.root == pathlib.Path(
        '/tmp/exp/artifacts/ingested_data'
    )
    assert art.data_preparation.root == pathlib.Path(
        '/tmp/exp/artifacts/prepared_data'
    )
    assert art.session.root == pathlib.Path('/tmp/exp/results')


def test_harmonization_paths(tmp_path):
    '''
    Given: A harmonization root directory path.
    When: Constructing unbound and allocated `HarmonizationPaths` handles.
    Then: Keep `from_root` side-effect-free, guard unbound run properties,
        and auto-increment run IDs only on `allocate_run`.
    '''
    e = tmp_path
    unbound = paths_mod.HarmonizationPaths.from_root(root=e)
    assert not unbound.is_run_bound
    assert list(e.iterdir()) == []
    with pytest.raises(
        ctrl_mod.ArtifactError, match='handle is not bound to a run folder'
    ):
        _ = unbound.effective_run_folder

    paths = paths_mod.HarmonizationPaths.allocate_run(root=e)
    assert paths.is_run_bound
    paths.init_pipeline_folders()
    r = paths.effective_run_folder

    assert paths.run_id == 'run_0001'
    assert paths.valid_mask_raster == r / 'valid_pixel_mask.vrt'
    assert paths.config == r / 'config.json'
    assert paths.report == r / 'report.json'
    assert paths.runs_manifest.is_file()
    assert ctrl_mod.Controller[dict](paths.runs_manifest).fetch() == {}

    # second allocate_run auto-increments run_id
    etl_paths_2 = paths_mod.HarmonizationPaths.allocate_run(root=e)
    etl_paths_2.init_pipeline_folders()
    assert etl_paths_2.run_id == 'run_0002'


def test_harmonization_paths_open_and_list_runs(tmp_path):
    '''
    Given: Multiple harmonization run folders (run_0001, run_0002).
    When: Calling `list_runs`, `open_run`, and `from_run_folder`.
    Then: List sorted runs, resolve targeted or latest by numeric ID
        regardless of mtime, and perform zero allocations.
    '''
    root = pathlib.Path(tmp_path) / 'harm_root'
    run1 = root / 'run_0001'
    run2 = root / 'run_0002'
    run1.mkdir(parents=True, exist_ok=True)
    run2.mkdir(parents=True, exist_ok=True)

    # Touch run1 newer than run2 to ensure numeric ordering is used
    now = time.time()
    os.utime(run2, (now - 100, now - 100))
    os.utime(run1, (now + 100, now + 100))

    listed = paths_mod.HarmonizationPaths.list_runs(root)
    assert listed == ['run_0001', 'run_0002']

    # default (None) -> highest numeric ID (run_0002)
    latest = paths_mod.HarmonizationPaths.open_run(root)
    assert latest.effective_run_folder == run2

    # int index -> run_0001
    assert (
        paths_mod.HarmonizationPaths.open_run(root, 1).effective_run_folder
        == run1
    )

    # digit string -> run_0001
    assert (
        paths_mod.HarmonizationPaths.open_run(root, '1').effective_run_folder
        == run1
    )

    # folder name string -> run_0001
    assert (
        paths_mod.HarmonizationPaths.open_run(
            root, 'run_0001'
        ).effective_run_folder
        == run1
    )

    # direct path -> run_0001
    assert (
        paths_mod.HarmonizationPaths.open_run(root, run1).effective_run_folder
        == run1
    )

    # decoupled run folder -> bound without allocation
    decoupled = tmp_path / 'external_harm'
    decoupled.mkdir()
    bound = paths_mod.HarmonizationPaths.from_run_folder(decoupled)
    assert bound.effective_run_folder == decoupled
    assert sorted(p.name for p in root.iterdir()) == ['run_0001', 'run_0002']

    # empty root raises ArtifactError instead of mkdir
    empty = tmp_path / 'empty_harm'
    empty.mkdir()
    with pytest.raises(ctrl_mod.ArtifactError, match='No run folders found'):
        paths_mod.HarmonizationPaths.open_run(empty)
    assert list(empty.iterdir()) == []


def test_ingestion_paths(tmp_path):
    '''
    Given: An ingestion root directory string.
    When: Accessing `IngestionPaths` properties and helper methods.
    Then: Return expected report, config, and domain map filepaths.
    '''
    f = tmp_path / 'ingested_data'
    f_paths = paths_mod.IngestionPaths.allocate_run(root=f)
    f_paths.init_pipeline_folders()

    assert f_paths.report == f / 'run_0001' / 'report.json'
    assert f_paths.config == f / 'run_0001' / 'config.json'
    assert f_paths.runs_manifest.is_file()
    assert ctrl_mod.Controller[dict](f_paths.runs_manifest).fetch() == {}

    # domain maps fpaths
    d_dir = f / 'domain_knowledge'
    dom_map = f_paths.domains.domain_map_fpath('eco.tif')
    assert dom_map == d_dir / 'eco.json'

    mapped_tiles = f_paths.domains.mapped_tiles_fpath('eco.tif', 'gid1')
    expected_npz = 'eco_tiles_gid1.npz'
    assert mapped_tiles == d_dir / expected_npz


def test_data_blocks_paths():
    '''
    Given: An IngestionPaths instance.
    When: Accessing `data_blocks` paths.
    Then: Return expected canonical data block artifact paths.
    '''
    f = pathlib.Path('/tmp') / 'exp' / 'ingested_data'
    f_paths = paths_mod.IngestionPaths(root=f)
    db = f_paths.data_blocks
    b = f / 'data_blocks'

    assert db.blocks == b / 'blocks'
    assert db.catalog == b / 'catalog.json'
    assert db.schema == b / 'schema.json'


# ----- `PreparationPaths` tests
def test_preparation_paths():
    '''
    Given: A preparation root directory string.
    When: Accessing `PreparationPaths` property endpoints.
    Then: Return canonical file and folder paths for prepared datasets.
    '''
    t = pathlib.Path('/tmp') / 'exp' / 'prepared_data'
    t_paths = paths_mod.PreparationPaths(root=t)

    assert t_paths.report == t / 'report.json'
    assert t_paths.config == t / 'config.json'
    assert t_paths.train_blocks == t / 'train_blocks'
    assert t_paths.val_blocks == t / 'val_blocks'
    assert t_paths.test_blocks == t / 'test_blocks'
    assert t_paths.splits_source_blocks == t / 'block_splits_source.json'
    assert t_paths.splits_summary == t / 'block_splits_summary.json'
    assert t_paths.label_stats == t / 'label_stats.json'
    assert t_paths.image_stats == t / 'image_stats.json'
    assert t_paths.splits_prepared_blocks == t / 'block_splits_prepared.json'
    assert t_paths.schema == t / 'schema.json'


# ----- `SessionPaths` tests
def test_session_paths_init_and_checkpoints(tmp_path):
    '''
    Given: A results root directory path.
    When: Using unbound, allocated, opened, and decoupled `SessionPaths`.
    Then: Keep `from_root` side-effect-free, auto-increment on allocation,
        and build all session artifact and visual filepaths.
    '''
    unbound = paths_mod.SessionPaths.from_root(tmp_path)
    assert not unbound.is_run_bound
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(
        ctrl_mod.ArtifactError, match='handle is not bound to a run folder'
    ):
        _ = unbound.checkpoints
    with pytest.raises(
        ctrl_mod.ArtifactError, match='handle is not bound to a run folder'
    ):
        _ = unbound.training_curves

    session_paths = paths_mod.SessionPaths.allocate_run(root=tmp_path)
    session_paths.init_pipeline_folders()

    assert session_paths.run_id == 'run_0001'
    assert session_paths.checkpoints.is_dir()
    assert session_paths.logs.is_dir()
    assert session_paths.plots.is_dir()
    assert session_paths.previews.is_dir()

    c = session_paths.checkpoints
    p = session_paths.plots
    v = session_paths.previews
    r = session_paths.effective_run_folder
    assert session_paths.best_checkpoint('model') == c / 'model_best.pt'
    assert session_paths.last_checkpoint('model') == c / 'model_last.pt'
    assert session_paths.checkpoint_catalog == c / 'status.json'
    assert session_paths.training_curves == p / 'training_curves.png'
    assert session_paths.per_class_plot('lc') == p / 'lc_per_class_iou.png'
    assert (
        session_paths.confusion_matrix_plot('lc')
        == p / 'lc_confusion_matrix.png'
    )
    assert session_paths.preview('lc') == v / 'lc_preview.png'
    assert session_paths.config == r / 'config.json'
    assert session_paths.report == r / 'report.json'

    (v / 'b_preview.png').write_bytes(b'b')
    (v / 'a_preview.png').write_bytes(b'a')
    assert session_paths.list_previews() == [
        v / 'a_preview.png',
        v / 'b_preview.png',
    ]

    # second run allocation auto-increments run_id
    results_2 = paths_mod.SessionPaths.allocate_run(root=tmp_path)
    results_2.init_pipeline_folders()
    assert results_2.run_id == 'run_0002'

    # open_run selects highest numeric run without allocation
    results_last = paths_mod.SessionPaths.open_run(root=tmp_path)
    assert results_last.run_id == 'run_0002'

    # from_run_folder accepts non-canonical standalone run folders
    standalone = tmp_path / 'standalone_trial'
    standalone.mkdir()
    bound_standalone = paths_mod.SessionPaths.from_run_folder(standalone)
    assert bound_standalone.effective_run_folder == standalone
    assert bound_standalone.run_id == 'standalone_trial'


def test_resolve_session_run(tmp_path):
    '''
    Given: Experiment root, results root, canonical run folder, decoupled
        run folder, and invalid foreign directories.
    When: Calling `resolve_session_run`.
    Then: Classify each input by schema markers, resolve target or latest
        numeric run without side effects, and reject invalid directories.
    '''
    exp_root = tmp_path / 'exp'
    results_root = exp_root / 'results'
    run1 = paths_mod.SessionPaths.allocate_run(results_root).init_pipeline_folders()
    run2 = paths_mod.SessionPaths.allocate_run(results_root).init_pipeline_folders()

    # Make run_0001 newer on disk so mtime-based discovery would be wrong
    now = time.time()
    os.utime(run2.effective_run_folder, (now - 100, now - 100))
    os.utime(run1.effective_run_folder, (now + 100, now + 100))

    # 1. Experiment root -> latest numeric run (run_0002) and targeted run
    resolved_exp = paths_mod.resolve_session_run(exp_root)
    assert resolved_exp.effective_run_folder == run2.effective_run_folder
    assert (
        paths_mod.resolve_session_run(exp_root, run_id=1).effective_run_folder
        == run1.effective_run_folder
    )

    # 2. Results root -> latest numeric run and targeted run
    assert (
        paths_mod.resolve_session_run(results_root).effective_run_folder
        == run2.effective_run_folder
    )
    assert (
        paths_mod.resolve_session_run(
            results_root, run_id='run_0001'
        ).effective_run_folder
        == run1.effective_run_folder
    )

    # 3. Canonical run folder -> self
    assert (
        paths_mod.resolve_session_run(
            run1.effective_run_folder
        ).effective_run_folder
        == run1.effective_run_folder
    )
    with pytest.raises(ctrl_mod.ArtifactError, match='does not match'):
        paths_mod.resolve_session_run(run1.effective_run_folder, run_id='run_0002')

    # 4. Decoupled run folder with session marker
    decoupled = tmp_path / 'optuna_trial_0003'
    (decoupled / 'checkpoints').mkdir(parents=True)
    assert (
        paths_mod.resolve_session_run(decoupled).effective_run_folder
        == decoupled
    )

    # 5. Foreign / missing directories raise ArtifactError with no side effects
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    with pytest.raises(
        ctrl_mod.ArtifactError, match='Unrecognized session directory layout'
    ):
        paths_mod.resolve_session_run(foreign)
    assert list(foreign.iterdir()) == []

    missing = tmp_path / 'does_not_exist'
    with pytest.raises(ctrl_mod.ArtifactError, match='does not exist'):
        paths_mod.resolve_session_run(missing)
    assert not missing.exists()


def test_artifact_paths_from_exp_root(tmp_path):
    '''
    Given: An experiment root path.
    When: Calling `ArtifactPaths.from_exp_root`.
    Then: Populate stage roots beneath `<exp_root>/artifacts` and
        `<exp_root>/results` without creating directories.
    '''
    exp = tmp_path / 'my_exp'
    art = paths_mod.ArtifactPaths.from_exp_root(exp)
    assert art.world_grid.root == exp / 'artifacts' / 'world_grids'
    assert art.data_harmonization.root == exp / 'artifacts' / 'harmonized_data'
    assert art.session.root == exp / 'results'
    assert not exp.exists()


# ----- `KnowledgePaths` tests
def test_knowledge_paths_defaults_and_fpaths():
    '''
    Given: A default or custom knowledge root directory string.
    When: Calling `KnowledgePaths` helper methods for a profile.
    Then: Return joined filepaths to matrix and manifest artifacts.
    '''
    kp = paths_mod.KnowledgePaths()
    assert kp.root == pathlib.Path('knowledge')
    assert kp.profile_dpath('my_profile') == (
        pathlib.Path('knowledge') / 'embeddings' / 'my_profile'
    )
    assert kp.similarity_matrix_fpath('my_profile') == (
        pathlib.Path('knowledge')
        / 'embeddings'
        / 'my_profile'
        / 'species_similarity_matrix.pt'
    )
    assert kp.embeddings_fpath('my_profile') == (
        pathlib.Path('knowledge')
        / 'embeddings'
        / 'my_profile'
        / 'species_embeddings.pt'
    )
    assert kp.metadata_fpath('my_profile') == (
        pathlib.Path('knowledge')
        / 'embeddings'
        / 'my_profile'
        / 'species_metadata.json'
    )
    assert kp.similarity_csv_fpath('my_profile') == (
        pathlib.Path('knowledge')
        / 'embeddings'
        / 'my_profile'
        / 'species_similarity_matrix.csv'
    )

    # custom knowledge root
    kp_custom = paths_mod.KnowledgePaths(root=pathlib.Path('/opt/knowledge'))
    assert kp_custom.similarity_matrix_fpath('tree_prof') == (
        pathlib.Path('/opt/knowledge')
        / 'embeddings'
        / 'tree_prof'
        / 'species_similarity_matrix.pt'
    )


def test_artifact_paths_knowledge_integration():
    '''
    Given: `ArtifactPaths` with default or overridden knowledge_root.
    When: Accessing `.knowledge` property.
    Then: Return configured `KnowledgePaths` instance.
    '''
    art = paths_mod.ArtifactPaths(knowledge_root=pathlib.Path('/data/knowledge'))
    assert isinstance(art.knowledge, paths_mod.KnowledgePaths)
    assert art.knowledge.root == pathlib.Path('/data/knowledge')

    art_default = paths_mod.ArtifactPaths()
    assert art_default.knowledge is None


# ----- `validate` method tests for all path classes
def test_pipeline_artifacts_paths_validate(tmp_path):
    '''
    Given: A Base PipelineArtifactsPaths instance.
    When: Calling `validate`.
    Then: Succeed if directory exists; raise ArtifactError if missing.
    '''
    valid_dir = tmp_path / 'valid'
    valid_dir.mkdir()
    p = paths_mod.PipelineArtifactsPaths(root=valid_dir)
    assert p.validate() is p

    missing_dir = tmp_path / 'missing'
    p_missing = paths_mod.PipelineArtifactsPaths(root=missing_dir)
    with pytest.raises(ctrl_mod.ArtifactError, match='does not exist'):
        p_missing.validate()

    file_as_root = tmp_path / 'file_root'
    file_as_root.write_text('content')
    p_file = paths_mod.PipelineArtifactsPaths(root=file_as_root)
    with pytest.raises(ctrl_mod.ArtifactError, match='not a directory'):
        p_file.validate()


def test_world_grid_paths_validate(tmp_path):
    '''
    Given: A WorldGridPaths instance.
    When: Calling `validate`.
    Then: Succeed if report.json exists; raise ArtifactError otherwise.
    '''
    grid_root = tmp_path / 'world_grid'
    grid_root.mkdir()
    paths = paths_mod.WorldGridPaths(root=grid_root)

    with pytest.raises(ctrl_mod.ArtifactError, match='report does not exist'):
        paths.validate()

    paths.report.write_text('{}')
    assert paths.validate() is paths


def test_harmonization_paths_validate(tmp_path):
    '''
    Given: A HarmonizationPaths instance.
    When: Calling `validate` with default and run requirements.
    Then: Validate presence of runs_manifest and bound run folder contents.
    '''
    harm_root = tmp_path / 'harmonization'
    harm_root.mkdir()
    unbound = paths_mod.HarmonizationPaths.from_root(root=harm_root)

    with pytest.raises(ctrl_mod.ArtifactError, match='manifest does not exist'):
        unbound.validate()

    paths = paths_mod.HarmonizationPaths.allocate_run(root=harm_root)
    paths.init_pipeline_folders()
    assert paths.validate() is paths
    assert unbound.validate() is unbound

    with pytest.raises(
        ctrl_mod.ArtifactError, match='handle is not bound to a run folder'
    ):
        unbound.validate(require_run=True)

    with pytest.raises(ctrl_mod.ArtifactError, match='valid mask raster'):
        paths.validate(require_run=True)

    paths.valid_mask_raster.write_text('vrt')
    assert paths.validate(require_run=True) is paths


def test_ingestion_paths_validate(tmp_path):
    '''
    Given: An IngestionPaths instance.
    When: Calling `validate`.
    Then: Enforce data blocks schema, domains, blocks pool, and
        manifest.
    '''
    ing_root = tmp_path / 'ingestion'
    ing_root.mkdir()
    paths = paths_mod.IngestionPaths(root=ing_root)

    with pytest.raises(ctrl_mod.ArtifactError, match='Data blocks directory'):
        paths.validate()

    paths.data_blocks.root.mkdir(parents=True)
    with pytest.raises(ctrl_mod.ArtifactError, match='schema file'):
        paths.validate()

    paths.data_blocks.schema.write_text('{}')
    with pytest.raises(ctrl_mod.ArtifactError, match='Domain knowledge directory'):
        paths.validate()

    paths.domains.root.mkdir(parents=True)
    assert paths.validate() is paths

    # require blocks
    with pytest.raises(ctrl_mod.ArtifactError, match='Data blocks pool directory'):
        paths.validate(require_blocks=True)
    paths.data_blocks.blocks.mkdir(parents=True)
    assert paths.validate(require_blocks=True) is paths

    # require manifest
    with pytest.raises(ctrl_mod.ArtifactError, match='manifest does not exist'):
        paths.validate(require_manifest=True)
    paths.runs_manifest.write_text('{}')
    assert paths.validate(require_manifest=True) is paths


def test_preparation_paths_validate(tmp_path):
    '''
    Given: A PreparationPaths instance.
    When: Calling `validate`.
    Then: Validate schema and optional split block directories and
        manifest.
    '''
    prep_root = tmp_path / 'preparation'
    prep_root.mkdir()
    paths = paths_mod.PreparationPaths(root=prep_root)

    with pytest.raises(ctrl_mod.ArtifactError, match='Preparation schema file'):
        paths.validate()

    paths.schema.write_text('{}')
    assert paths.validate() is paths

    # require splits
    with pytest.raises(ctrl_mod.ArtifactError, match='Train blocks directory'):
        paths.validate(require_splits=True)

    paths.train_blocks.mkdir()
    paths.val_blocks.mkdir()
    paths.test_blocks.mkdir()
    with pytest.raises(ctrl_mod.ArtifactError, match='Prepared block splits manifest'):
        paths.validate(require_splits=True)

    paths.splits_prepared_blocks.write_text('{}')
    assert paths.validate(require_splits=True) is paths


def test_session_paths_validate(tmp_path):
    '''
    Given: Unbound and bound `SessionPaths` instances.
    When: Calling `validate` with default and run requirements.
    Then: Validate root and session run directory structure.
    '''
    sess_root = tmp_path / 'results'
    sess_root.mkdir()
    unbound = paths_mod.SessionPaths.from_root(root=sess_root)
    assert unbound.validate() is unbound

    with pytest.raises(
        ctrl_mod.ArtifactError, match='handle is not bound to a run folder'
    ):
        unbound.validate(require_run=True)

    bound = paths_mod.SessionPaths.allocate_run(root=sess_root)
    with pytest.raises(ctrl_mod.ArtifactError, match='Checkpoints directory'):
        bound.validate(require_run=True)

    bound.init_pipeline_folders()
    assert bound.validate(require_run=True) is bound


def test_knowledge_paths_validate(tmp_path):
    '''
    Given: A KnowledgePaths instance.
    When: Calling `validate`.
    Then: Validate root and species profile embeddings/matrices.
    '''
    k_root = tmp_path / 'knowledge'
    paths = paths_mod.KnowledgePaths(root=k_root)

    with pytest.raises(ctrl_mod.ArtifactError, match='Knowledge root directory'):
        paths.validate()

    k_root.mkdir()
    assert paths.validate() is paths

    with pytest.raises(ctrl_mod.ArtifactError, match='Profile directory'):
        paths.validate(profile='boreal')

    prof_dir = paths.profile_dpath('boreal')
    prof_dir.mkdir(parents=True)
    with pytest.raises(ctrl_mod.ArtifactError, match='similarity matrix'):
        paths.validate(profile='boreal')

    paths.similarity_matrix_fpath('boreal').write_text('matrix')
    with pytest.raises(ctrl_mod.ArtifactError, match='embeddings do not exist'):
        paths.validate(profile='boreal')

    paths.embeddings_fpath('boreal').write_text('embeddings')
    with pytest.raises(ctrl_mod.ArtifactError, match='metadata does not exist'):
        paths.validate(profile='boreal')

    paths.metadata_fpath('boreal').write_text('{}')
    assert paths.validate(profile='boreal') is paths


def test_artifact_paths_validate(tmp_path):
    '''
    Given: An ArtifactPaths aggregation instance.
    When: Calling `validate`.
    Then: Validate configured stage directories and stage selection.
    '''
    exp_root = tmp_path / 'exp'
    art = paths_mod.ArtifactPaths(root=exp_root)

    # Missing root throws on first stage
    with pytest.raises(ctrl_mod.ArtifactError):
        art.validate()

    # Create directories and files for world_grid and harmonization
    art.world_grid.root.mkdir(parents=True)
    art.world_grid.report.write_text('{}')
    paths_mod.HarmonizationPaths.allocate_run(
        art.data_harmonization.root
    ).init_pipeline_folders()

    # Validating selected valid stages succeeds
    assert art.validate(stages=['world_grid', 'data_harmonization']) is art

    # Invalid stage name raises ValueError
    with pytest.raises(ValueError, match='Unknown stage name'):
        art.validate(stages=['invalid_stage'])
