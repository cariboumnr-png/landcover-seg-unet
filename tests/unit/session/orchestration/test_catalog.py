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

'''Unit tests for `landseg.session.orchestration.catalog`.'''

# standard imports
import json
import pathlib
import shutil
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.session.orchestration.catalog as catalog


def _make_run(root: pathlib.Path) -> artifacts.SessionPaths:
    '''Allocate and initialize a session run folder under `root`.'''
    run = artifacts.SessionPaths.allocate_run(root)
    run.init_pipeline_folders()
    return run


# ----- round-trip and relocation tests
def test_catalog_round_trip_preserves_best_and_relocates(tmp_path):
    '''
    Given: A phase with an early best checkpoint and a later non-best save.
    When: Persisting, reloading, and moving the run folder elsewhere.
    Then: Keep best metadata, store relative paths, and resolve after move.
    '''
    run = _make_run(tmp_path / 'results')
    best_pt = run.best_checkpoint('base_epoch')
    last_pt = run.last_checkpoint('base_epoch')
    best_pt.write_bytes(b'best')
    last_pt.write_bytes(b'last')

    cat = catalog.CheckpointCatalog(run=run)
    cat.record_save(
        'base',
        last_fpath=last_pt,
        epoch=1,
        best_fpath=best_pt,
        metric=0.72,
    )
    cat.record_save(
        'base',
        last_fpath=last_pt,
        epoch=2,
        best_fpath=None,
        metric=0.68,
    )
    cat.mark_phase_completed('base')
    cat.persist()

    raw = json.loads(run.checkpoint_catalog.read_text(encoding='utf-8'))
    assert raw['schema_id'] == catalog.SCHEMA_ID
    assert raw['policy'] == 'last_phase_best'
    assert raw['phases']['base'] == {
        'best_fpath': 'checkpoints/base_epoch_best.pt',
        'last_fpath': 'checkpoints/base_epoch_last.pt',
        'best_epoch': 1,
        'best_metric': 0.72,
        'last_epoch': 2,
        'completed': True,
    }

    reloaded = catalog.CheckpointCatalog.from_run(run)
    assert reloaded.best('base') == best_pt
    assert reloaded.last('base') == last_pt
    assert reloaded.final == best_pt

    moved_dir = tmp_path / 'moved_run'
    shutil.move(str(run.effective_run_folder), str(moved_dir))
    moved_run = artifacts.SessionPaths.from_run_folder(moved_dir)
    moved_cat = catalog.CheckpointCatalog.from_run(moved_run)
    assert moved_cat.final == moved_dir / 'checkpoints' / 'base_epoch_best.pt'
    assert moved_cat.best('base') == moved_cat.final


# ----- multi-phase selection policy tests
def test_catalog_multi_phase_uses_last_phase_best(tmp_path):
    '''
    Given: A curriculum catalog where phase 1 metric exceeds phase 2 metric.
    When: Querying `catalog.final`.
    Then: Select phase 2 best checkpoint under policy `last_phase_best`.
    '''
    run = _make_run(tmp_path / 'results')
    p1_best = run.best_checkpoint('freeze_epoch')
    p1_last = run.last_checkpoint('freeze_epoch')
    p2_best = run.best_checkpoint('finetune_epoch')
    p2_last = run.last_checkpoint('finetune_epoch')
    for fp in (p1_best, p1_last, p2_best, p2_last):
        fp.write_bytes(b'pt')

    cat = catalog.CheckpointCatalog(run=run)
    cat.record_save(
        'freeze', last_fpath=p1_last, epoch=3, best_fpath=p1_best, metric=0.91
    )
    cat.mark_phase_completed('freeze')
    cat.record_save(
        'finetune', last_fpath=p2_last, epoch=2, best_fpath=p2_best, metric=0.64
    )
    cat.mark_phase_completed('finetune')
    cat.persist()

    loaded = catalog.CheckpointCatalog.from_run(run)
    assert loaded.policy == 'last_phase_best'
    assert loaded.final == p2_best
    assert loaded.best('freeze') == p1_best
    assert loaded.last('finetune') == p2_last


def test_catalog_falls_back_to_prior_phase_and_validates_files(tmp_path):
    '''
    Given: Last phase saved no best (`-inf` metric sanitized) after phase 1.
    When: Reading `final` or querying a deleted/missing checkpoint file.
    Then: Fall back to prior phase when `best_fpath` is None and raise
        `ArtifactError` if any referenced `.pt` is missing or unknown.
    '''
    run = _make_run(tmp_path / 'results')
    p1_best = run.best_checkpoint('p1_epoch')
    p1_last = run.last_checkpoint('p1_epoch')
    p2_last = run.last_checkpoint('p2_epoch')
    for fp in (p1_best, p1_last, p2_last):
        fp.write_bytes(b'pt')

    cat = catalog.CheckpointCatalog(run=run)
    cat.record_save(
        'p1', last_fpath=p1_last, epoch=1, best_fpath=p1_best, metric=0.55
    )
    cat.record_save(
        'p2',
        last_fpath=p2_last,
        epoch=1,
        best_fpath=None,
        metric=-float('inf'),
    )
    cat.persist()

    loaded = catalog.CheckpointCatalog.from_run(run)
    assert loaded.phases['p2'].best_metric is None
    assert loaded.final == p1_best

    with pytest.raises(artifacts.ArtifactError, match='not found'):
        loaded.best('unknown_phase')
    with pytest.raises(artifacts.ArtifactError, match='No best checkpoint'):
        loaded.best('p2')

    p1_best.unlink()
    with pytest.raises(artifacts.ArtifactError, match='does not exist'):
        _ = loaded.final


# ----- legacy fallback tests
def test_catalog_legacy_fallback_handles_zero_one_and_many(tmp_path):
    '''
    Given: Legacy runs without `checkpoints/status.json`.
    When: Calling `CheckpointCatalog.from_run` with 0, 1, or >1 `*_best.pt`.
    Then: Return single legacy checkpoint cleanly and fail on 0 or >1.
    '''
    run_empty = _make_run(tmp_path / 'empty')
    with pytest.raises(artifacts.ArtifactError, match='No checkpoint catalog'):
        catalog.CheckpointCatalog.from_run(run_empty)

    run_single = _make_run(tmp_path / 'single')
    legacy_best = run_single.checkpoints / 'model_epoch_best.pt'
    legacy_best.write_bytes(b'legacy')
    legacy_cat = catalog.CheckpointCatalog.from_run(run_single)
    assert legacy_cat.final == legacy_best
    assert legacy_cat.best('model') == legacy_best

    run_multi = _make_run(tmp_path / 'multi')
    (run_multi.checkpoints / 'phase1_epoch_best.pt').write_bytes(b'1')
    (run_multi.checkpoints / 'phase2_epoch_best.pt').write_bytes(b'2')
    with pytest.raises(artifacts.ArtifactError, match='Ambiguous legacy checkpoints'):
        catalog.CheckpointCatalog.from_run(run_multi)
