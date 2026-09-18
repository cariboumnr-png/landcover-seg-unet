# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''
Unit tests for `landseg.artifacts.paths`.
'''

# standard imports
import os
# local imports
import landseg.artifacts.paths as paths_mod
import landseg.artifacts.paths.data_ingestion as f_mod
import landseg.configs as configs_mod


# ----- `ArtifactPaths` & Foundation tests
def test_artifact_paths_hierarchy():
    '''
    Given: An experiment root directory string.
    When: Instantiating `ArtifactPaths`.
    Then: Return joined namespaces.
    '''
    r = os.path.join('/tmp', 'exp')
    art = paths_mod.ArtifactPaths(root=r)
    assert art.data_harmonization.root == os.path.join(r, 'harmonized_data')
    assert art.data_ingestion.root == os.path.join(r, 'ingested_data')
    assert art.data_preparation.root == os.path.join(r, 'prepared_data')
    assert art.session.root == os.path.join(r, 'results')


def test_artifact_paths_custom_overrides():
    '''
    Given: Explicit per-pipeline sub-root path overrides.
    When: Instantiating `ArtifactPaths` with sub-root parameters.
    Then: Prioritize custom sub-root paths over central exp_root defaults.
    '''
    art = paths_mod.ArtifactPaths(
        root='/tmp/exp',
        harmonization_root='/custom/harmonized',
        ingestion_root='/custom/ingested',
        preparation_root='/custom/prepared',
        session_root='/custom/session'
    )
    assert art.data_harmonization.root == '/custom/harmonized'
    assert art.data_ingestion.root == '/custom/ingested'
    assert art.data_preparation.root == '/custom/prepared'
    assert art.session.root == '/custom/session'


def test_artifact_paths_from_config():
    '''
    Given: A resolved `RootConfig` instance with explicit output directories.
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
    assert art.root == '/tmp/exp'
    assert art.data_harmonization.root == '/tmp/exp/artifacts/harmonized_data'
    assert art.data_ingestion.root == '/tmp/exp/artifacts/ingested_data'
    assert art.data_preparation.root == '/tmp/exp/artifacts/prepared_data'
    assert art.session.root == '/tmp/exp/results'


def test_harmonization_paths(tmp_path):
    '''
    Given: A harmonization root directory path.
    When: Initializing `HarmonizationPaths` across runs.
    Then: Return expected run-isolated valid mask raster, config, and
        harmonize_report.json file paths.
    '''
    e = str(tmp_path)
    paths = paths_mod.HarmonizationPaths(root=e)
    paths.init()

    assert paths.run_id == 'run_0001'
    r = paths.effective_root
    assert paths.valid_mask_raster == os.path.join(r, 'valid_pixel_mask.vrt')
    assert paths.config == os.path.join(r, 'config.json')
    assert paths.report == os.path.join(r, 'harmonize_report.json')

    # second init auto-increments run_id
    etl_paths_2 = paths_mod.HarmonizationPaths(root=e)
    etl_paths_2.init()
    assert etl_paths_2.run_id == 'run_0002'


def test_harmonization_paths_get_run_folder(tmp_path):
    '''
    Given: Multiple harmonization run folders (run_0001, run_0002).
    When: Calling `get_run_folder` with ints, folder strings, or paths.
    Then: Resolve targeted folder or default to latest run.
    '''
    root = str(tmp_path)
    run1 = os.path.join(root, 'run_0001')
    run2 = os.path.join(root, 'run_0002')
    os.makedirs(run1, exist_ok=True)
    os.makedirs(run2, exist_ok=True)

    h_paths = paths_mod.HarmonizationPaths(root=root)

    # default (None) -> latest (run_0002)
    assert h_paths.get_run_folder() == run2

    # int index -> run_0001
    assert h_paths.get_run_folder(1) == run1

    # digit string -> run_0001
    assert h_paths.get_run_folder('1') == run1

    # folder name string -> run_0001
    assert h_paths.get_run_folder('run_0001') == run1

    # direct path string -> run_0001
    assert h_paths.get_run_folder(run1) == run1


def test_ingestion_paths():
    '''
    Given: An ingestion root directory string.
    When: Accessing `IngestionPaths` properties and sub-container helpers.
    Then: Return expected report, config, and domain map filepaths.
    '''
    f = os.path.join('/tmp', 'exp', 'ingested_data')
    f_paths = paths_mod.IngestionPaths(root=f)
    assert f_paths.report == os.path.join(f, 'ingest_report.json')
    assert f_paths.config == os.path.join(f, 'config.json')

    # domain maps fpaths
    d_dir = os.path.join(f, 'domain_knowledge')
    dom_map = f_paths.domains.domain_map_fpath('eco.tif')
    assert dom_map == os.path.join(d_dir, 'eco.json')

    mapped_tiles = f_paths.domains.mapped_tiles_fpath('eco.tif', 'gid1')
    expected_npz = 'eco_tiles_gid1.npz'
    assert mapped_tiles == os.path.join(d_dir, expected_npz)


def test_data_blocks_paths():
    '''
    Given: A data blocks root directory string.
    When: Accessing `_DataBlocks` paths and window map methods.
    Then: Return expected canonical data block artifact paths.
    '''
    b = os.path.join('/tmp', 'exp', 'ingested_data', 'data_blocks')
    db = f_mod._DataBlocks(root=b)

    assert db.blocks == os.path.join(b, 'blocks')
    assert db.catalog == os.path.join(b, 'catalog.json')
    assert db.schema == os.path.join(b, 'schema.json')

    win_path = db.mapped_window('g1')
    assert win_path == os.path.join(db.windows, 'windows_g1.json')



# ----- `PreparationPaths` tests
def test_preparation_paths():
    '''
    Given: A preparation root directory string.
    When: Accessing `PreparationPaths` property endpoints.
    Then: Return canonical file and folder paths for prepared datasets.
    '''
    t = os.path.join('/tmp', 'exp', 'prepared_data')
    t_paths = paths_mod.PreparationPaths(root=t)

    assert t_paths.report == os.path.join(t, 'prep_report.json')
    assert t_paths.config == os.path.join(t, 'config.json')
    assert t_paths.train_blocks == os.path.join(t, 'train_blocks')
    assert t_paths.val_blocks == os.path.join(t, 'val_blocks')
    assert t_paths.test_blocks == os.path.join(t, 'test_blocks')
    assert t_paths.splits_source_blocks == os.path.join(
        t, 'block_splits_source.json'
    )
    assert t_paths.splits_summary == os.path.join(t, 'block_splits_summary.json')
    assert t_paths.label_stats == os.path.join(t, 'label_stats.json')
    assert t_paths.image_stats == os.path.join(t, 'image_stats.json')
    assert t_paths.splits_prepared_blocks == os.path.join(
        t, 'block_splits_prepared.json'
    )
    assert t_paths.schema == os.path.join(t, 'schema.json')


# ----- `SessionPaths` tests
def test_session_paths_init_and_checkpoints(tmp_path):
    '''
    Given: A results root directory path.
    When: Initializing `SessionPaths` across runs and tracing options.
    Then: Auto-increment run IDs and build checkpoint paths.
    '''
    session_paths = paths_mod.SessionPaths(root=str(tmp_path))
    session_paths.init()

    assert session_paths.run_id == 'run_0001'
    assert os.path.isdir(session_paths.checkpoints)
    assert os.path.isdir(session_paths.logs)
    assert os.path.isdir(session_paths.plots)
    assert os.path.isdir(session_paths.previews)

    c = session_paths.checkpoints
    r = session_paths.run_folder
    assert session_paths.best_checkpoint('model') == os.path.join(c, 'model_best.pt')
    assert session_paths.last_checkpoint('model') == os.path.join(c, 'model_last.pt')
    assert session_paths.phase_status == os.path.join(c, 'status.json')
    assert session_paths.config == os.path.join(r, 'config.json')

    # second run initialization auto-increments run_id
    results_2 = paths_mod.SessionPaths(root=str(tmp_path))
    results_2.init()
    assert results_2.run_id == 'run_0002'

    # trace_to_last option targets previous run
    results_last = paths_mod.SessionPaths(root=str(tmp_path))
    results_last.init(trace_to_last=True)
    assert results_last.run_id == 'run_0002'


# ----- `KnowledgePaths` tests
def test_knowledge_paths_defaults_and_fpaths():
    '''
    Given: A default or custom knowledge root directory string.
    When: Calling `KnowledgePaths` helper methods for a profile.
    Then: Return joined filepaths to matrix and manifest artifacts.
    '''
    kp = paths_mod.KnowledgePaths()
    assert kp.root == 'knowledge'
    assert kp.profile_dpath('my_profile') == os.path.join(
        'knowledge', 'embeddings', 'my_profile'
    )
    assert kp.similarity_matrix_fpath('my_profile') == os.path.join(
        'knowledge', 'embeddings', 'my_profile',
        'species_similarity_matrix.pt'
    )
    assert kp.embeddings_fpath('my_profile') == os.path.join(
        'knowledge', 'embeddings', 'my_profile',
        'species_embeddings.pt'
    )
    assert kp.metadata_fpath('my_profile') == os.path.join(
        'knowledge', 'embeddings', 'my_profile',
        'species_metadata.json'
    )
    assert kp.similarity_csv_fpath('my_profile') == os.path.join(
        'knowledge', 'embeddings', 'my_profile',
        'species_similarity_matrix.csv'
    )

    # custom knowledge root
    kp_custom = paths_mod.KnowledgePaths(root='/opt/knowledge')
    assert kp_custom.similarity_matrix_fpath('tree_prof') == os.path.join(
        '/opt/knowledge', 'embeddings', 'tree_prof',
        'species_similarity_matrix.pt'
    )


def test_artifact_paths_knowledge_integration():
    '''
    Given: `ArtifactPaths` with default or overridden knowledge_root.
    When: Accessing `.knowledge` property.
    Then: Return configured `KnowledgePaths` instance.
    '''
    art = paths_mod.ArtifactPaths(knowledge_root='/data/knowledge')
    assert isinstance(art.knowledge, paths_mod.KnowledgePaths)
    assert art.knowledge.root == '/data/knowledge'
