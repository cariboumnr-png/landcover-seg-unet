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

'''Unit test suite for local scratch storage and staged publishing.'''

# standard imports
import os
import time
# third-party imports
import pytest
# local imports
from landseg.adapters.api import configurators
from landseg import artifacts
from landseg.artifacts.storage import scratch


# ----- `get_scratch_dir` tests
def test_get_scratch_dir_default():
    '''
    Given: Unconfigured environment without explicit arguments.
    When: Resolving the scratch directory.
    Then: Return a valid writable path inside system temp.
    '''
    sdir = scratch.get_scratch_dir()
    assert sdir.is_dir()
    assert sdir.name == 'landseg_scratch'
    assert os.access(sdir, os.W_OK)


def test_get_scratch_dir_custom_arg(tmp_path):
    '''
    Given: An explicit custom directory path.
    When: Resolving the scratch directory.
    Then: Return the resolved custom path and create it.
    '''
    custom = tmp_path / 'custom_scratch'
    sdir = scratch.get_scratch_dir(custom)
    assert sdir == custom.resolve()
    assert sdir.is_dir()


def test_get_scratch_dir_env_var(tmp_path, monkeypatch):
    '''
    Given: LANDSEG_SCRATCH_DIR set in environment variables.
    When: Resolving the scratch directory without arguments.
    Then: Return the path designated by the environment variable.
    '''
    env_dir = tmp_path / 'env_scratch'
    monkeypatch.setenv('LANDSEG_SCRATCH_DIR', str(env_dir))
    sdir = scratch.get_scratch_dir()
    assert sdir == env_dir.resolve()
    assert sdir.is_dir()


def test_get_scratch_dir_non_writable(tmp_path, monkeypatch):
    '''
    Given: A target path that is not writable.
    When: Resolving the scratch directory.
    Then: Raise PermissionError.
    '''
    target = tmp_path / 'readonly_scratch'
    target.mkdir()
    monkeypatch.setattr(os, 'access', lambda path, mode: False)
    with pytest.raises(PermissionError, match='not writable'):
        scratch.get_scratch_dir(target)


# ----- `scratch_file` tests
def test_scratch_file_lifecycle(tmp_path):
    '''
    Given: A scratch storage directory.
    When: Allocating a scratch_file context block.
    Then: Provide a unique temporary path and clean it up upon exit.
    '''
    with scratch.scratch_file(
        prefix='test_',
        suffix='.bin',
        scratch_dir=tmp_path
    ) as tmp_file:
        assert tmp_file.parent == tmp_path.resolve()
        assert tmp_file.name.startswith('test_')
        assert tmp_file.name.endswith('.bin')
        tmp_file.write_bytes(b'test_payload')
        assert tmp_file.exists()

    assert not tmp_file.exists()


def test_scratch_file_cleanup_on_exception(tmp_path):
    '''
    Given: An exception occurring inside a scratch_file block.
    When: Handling the exception.
    Then: Ensure the scratch file is cleaned up and not orphaned.
    '''
    leak_ref = None
    with pytest.raises(RuntimeError, match='boom'):
        with scratch.scratch_file(scratch_dir=tmp_path) as tmp_file:
            leak_ref = tmp_file
            tmp_file.write_text('should be cleaned up', encoding='UTF-8')
            raise RuntimeError('boom')

    assert leak_ref is not None
    assert not leak_ref.exists()


# ----- `stage_to_scratch` tests
def test_stage_to_scratch_publishes_on_success(tmp_path):
    '''
    Given: A destination file path and scratch directory.
    When: Writing to the staging file inside stage_to_scratch context.
    Then: Copy the staged file to destination and remove scratch copy.
    '''
    dest = tmp_path / 'dest' / 'published_artifact.tif'
    sdir = tmp_path / 'scratch'

    with scratch.stage_to_scratch(dest, scratch_dir=sdir) as staging:
        assert staging.parent == sdir.resolve()
        assert staging.name.startswith('.stage_')
        assert not dest.exists()
        staging.write_text('raster-payload-content', encoding='UTF-8')

    assert dest.exists()
    assert dest.read_text(encoding='UTF-8') == 'raster-payload-content'
    assert not staging.exists()


def test_stage_to_scratch_aborts_on_exception(tmp_path):
    '''
    Given: A failure occurring during staged generation.
    When: Catching the exception outside the context.
    Then: Leave destination absent and clean up staging path.
    '''
    dest = tmp_path / 'dest' / 'failed_artifact.tif'
    sdir = tmp_path / 'scratch'
    leak_ref = None

    with pytest.raises(ValueError, match='pipeline failure'):
        with scratch.stage_to_scratch(dest, scratch_dir=sdir) as staging:
            leak_ref = staging
            staging.write_text('incomplete-data', encoding='UTF-8')
            raise ValueError('pipeline failure')

    assert not dest.exists()
    assert leak_ref is not None
    assert not leak_ref.exists()


# ----- `cleanup_scratch` tests
def test_cleanup_scratch(tmp_path):
    '''
    Given: Multiple orphaned scratch files.
    When: Running cleanup_scratch.
    Then: Remove all files and return total removed count.
    '''
    (tmp_path / 'file_1.tmp').write_text('1', encoding='UTF-8')
    (tmp_path / 'file_2.tmp').write_text('2', encoding='UTF-8')
    (tmp_path / 'file_3.tmp').write_text('3', encoding='UTF-8')

    count = scratch.cleanup_scratch(tmp_path)
    assert count == 3
    assert not list(tmp_path.iterdir())


def test_cleanup_scratch_max_age(tmp_path):
    '''
    Given: A mix of old and freshly created scratch files.
    When: Running cleanup_scratch with max_age_seconds threshold.
    Then: Remove only files older than the specified age.
    '''
    old_file = tmp_path / 'old.tmp'
    new_file = tmp_path / 'new.tmp'
    old_file.write_text('old', encoding='UTF-8')
    new_file.write_text('new', encoding='UTF-8')

    # set old file modification time back by 100 seconds
    past = time.time() - 100
    os.utime(old_file, (past, past))

    count = scratch.cleanup_scratch(tmp_path, max_age_seconds=50)
    assert count == 1
    assert not old_file.exists()
    assert new_file.exists()


# ----- configurator integration test
def test_base_configurator_set_scratch(tmp_path):
    '''
    Given: A BaseConfigurator instance.
    When: Calling set_scratch with a valid directory.
    Then: Populate cfg.execution.scratch_dir.
    '''
    sdir = tmp_path / 'fast_scratch'
    sdir.mkdir()
    configurator = configurators.WorldGridConfigurator()
    configurator.set_scratch(sdir)
    assert configurator.cfg.execution.scratch_dir == str(sdir.resolve())


# ----- package export test
def test_artifacts_package_exports():
    '''
    Given: Top-level landseg.artifacts package.
    When: Accessing lazy storage exports.
    Then: Resolve all public scratch helpers.
    '''
    assert callable(artifacts.get_scratch_dir)
    assert callable(artifacts.scratch_file)
    assert callable(artifacts.stage_to_scratch)
    assert callable(artifacts.cleanup_scratch)
