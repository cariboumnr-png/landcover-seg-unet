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


'''
Unit tests for `landseg.artifacts.controller`.
'''

# third-party imports
import numpy
import pandas
import pytest
import torch
# local imports
import landseg.artifacts.controller as ctrl_mod
import landseg.artifacts.policy as policy_mod


# ----- `Controller` JSON tests
def test_json_persist_fetch_and_properties(tmp_path):
    '''
    Given: A target JSON file path and serializable dictionary payload.
    When: `Controller.persist()` and `Controller.fetch()` are executed.
    Then: Compute valid SHA-256 hash, update `_hash.json`, return data.
    '''
    json_path = str(tmp_path / 'data.json')
    ctrl = ctrl_mod.Controller[dict](json_path)

    assert ctrl.is_valid is False
    data_in = {'name': 'test_artifact', 'values': [1, 2, 3]}
    ctrl.persist(data_in)

    assert ctrl.is_valid is True
    assert len(ctrl.sha256) == 64
    assert isinstance(ctrl.creation_time, str)

    loaded = ctrl.fetch()
    assert loaded == data_in


# ----- `Controller` NPZ dict tests
def test_npz_dict_persist_and_fetch(tmp_path):
    '''
    Given: Target NPZ file path and array dictionary with tuple keys.
    When: `_npz_write_dict` and `_npz_read_dict` are invoked.
    Then: Save to compressed NPZ and reconstruct arrays with original keys.
    '''
    npz_path = str(tmp_path / 'tiles.npz')
    ctrl = ctrl_mod.Controller[dict](npz_path)

    arr1 = numpy.array([1, 2, 3])
    arr2 = numpy.array([4, 5, 6])
    data_in = {(0, 0): arr1, (0, 1): arr2}

    ctrl.persist(data_in)
    assert ctrl.is_valid is True

    loaded = ctrl.fetch()
    assert loaded is not None
    assert (0, 0) in loaded
    assert numpy.array_equal(loaded[(0, 0)], arr1)
    assert numpy.array_equal(loaded[(0, 1)], arr2)


def test_npz_write_validation(tmp_path):
    '''
    Given: An empty dictionary or invalid non-array dictionary values.
    When: Calling `Controller.persist()`.
    Then: Raise a ValueError indicating invalid input payload format.
    '''
    npz_path = str(tmp_path / 'invalid.npz')
    ctrl = ctrl_mod.Controller[dict](npz_path)

    with pytest.raises(ValueError, match='Cannot save empty dict'):
        ctrl.persist({})

    with pytest.raises(ValueError, match='Input source must be a dictionary'):
        ctrl.persist({'key': 'not_an_array'})  # type: ignore


# ----- `Controller` CSV tests
def test_csv_persist_fetch_and_hash(tmp_path):
    '''
    Given: Target CSV file path and DataFrame payload.
    When: `Controller.persist()` and `Controller.fetch()` are executed.
    Then: Persist to CSV, record hash, and load with integrity check.
    '''
    csv_path = str(tmp_path / 'table.csv')
    ctrl = ctrl_mod.Controller[pandas.DataFrame](csv_path)

    df_in = pandas.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    ctrl.persist(df_in)

    assert ctrl.is_valid is True
    loaded = ctrl.fetch()
    assert loaded is not None
    assert list(loaded['a']) == [1, 2]
    assert list(loaded['b']) == ['x', 'y']

    # default fetch succeeds
    loaded_ok = ctrl.fetch()
    assert loaded_ok is not None
    assert list(loaded_ok['a']) == [1, 2]

    # tampered CSV triggers hash mismatch
    with open(csv_path, 'a', encoding='UTF-8') as f:
        f.write('\n3,z')
    assert ctrl.is_valid is False
    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()


# ----- `Controller` PyTorch PT tests
def test_pt_persist_fetch_and_hash(tmp_path):
    '''
    Given: Target PyTorch PT file path and tensor payload.
    When: `Controller.persist()` and `Controller.fetch()` are executed.
    Then: Persist tensor, record hash, and load with integrity check.
    '''
    pt_path = str(tmp_path / 'tensor.pt')
    ctrl = ctrl_mod.Controller[torch.Tensor](pt_path)

    tensor_in = torch.randn(4, 4)
    ctrl.persist(tensor_in)

    assert ctrl.is_valid is True
    loaded = ctrl.fetch()
    assert loaded is not None
    assert torch.allclose(loaded, tensor_in)

    # tampered PT triggers hash mismatch
    tensor_tampered = torch.randn(4, 4)
    torch.save(tensor_tampered, pt_path)
    assert ctrl.is_valid is False
    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()

    # corrupted bytes trigger decode failure and ArtifactError
    with open(pt_path, 'wb') as f:
        f.write(b'corrupted_bytes')
    assert ctrl.is_valid is False
    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()


# ----- `Controller` lifecycle policies tests
def test_lifecycle_policies_fetch(tmp_path):
    '''
    Given: Artifact controllers configured with `LOAD_OR_FAIL`,
    `BUILD_IF_MISSING`, or `REBUILD`.
    When: Executing `get()` under non-existent or fresh state conditions.
    Then: Enforce policy rules (raise error, return None, or bypass).
    '''
    json_path = str(tmp_path / 'policy_test.json')

    # default policy is LOAD_OR_FAIL on missing file
    ctrl_default = ctrl_mod.Controller(json_path)
    assert ctrl_default.policy == policy_mod.LifecyclePolicy.LOAD_OR_FAIL
    with pytest.raises(
        ctrl_mod.ArtifactError, match='Required artifact is missing'
    ):
        ctrl_default.fetch()

    # missing_ok=True suppresses error and returns None
    assert ctrl_default.fetch(missing_ok=True) is None

    # explicit LOAD_OR_FAIL on missing file
    ctrl_fail = ctrl_mod.Controller(
        json_path,
        policy_mod.LifecyclePolicy.LOAD_OR_FAIL
    )
    with pytest.raises(
        ctrl_mod.ArtifactError, match='Required artifact is missing'
    ):
        ctrl_fail.fetch()

    # BUILD_IF_MISSING on missing file
    ctrl_build = ctrl_mod.Controller(
        file_path=json_path,
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )
    assert ctrl_build.fetch() is None

    # REBUILD policy on existing file
    ctrl_build.persist({'status': 'ok'})
    ctrl_rebuild = ctrl_mod.Controller(
        file_path=json_path,
        policy=policy_mod.LifecyclePolicy.REBUILD,
    )
    assert ctrl_rebuild.fetch() is None


# ----- Integrity mismatch tests
def test_controller_hash_mismatch_and_corruption(tmp_path):
    '''
    Given: An artifact file modified or corrupted on disk.
    When: Controller loads the artifact.
    Then: Detect hash mismatch or decode failure and raise `ArtifactError`.
    '''
    json_path = str(tmp_path / 'tampered.json')
    ctrl = ctrl_mod.Controller[dict](json_path)
    ctrl.persist({'key': 'original'})

    # modify file content without updating hash record
    with open(json_path, 'w', encoding='UTF-8') as f:
        f.write('{"key": "tampered"}')

    assert ctrl.is_valid is False
    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()

    # corrupt file contents with invalid JSON
    with open(json_path, 'w', encoding='UTF-8') as f:
        f.write('invalid json content...')

    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()


# ----- `Controller` schema validation tests
def test_controller_schema_validation(tmp_path):
    '''
    Given: An artifact with a schema_id bound at controller initialization.
    When: Loading and persisting payloads via fetch and persist.
    Then: Enforce schema matching on both read and write.
    '''
    fpath = tmp_path / 'payload.json'
    ctrl = ctrl_mod.Controller[dict](fpath, schema_id='test_schema/v0.1.0')

    # persisting mismatched schema raises ArtifactError
    with pytest.raises(ctrl_mod.ArtifactError, match='Schema mismatch'):
        ctrl.persist({'schema_id': 'wrong_schema/v0.1.0', 'data': [1]})

    ctrl.persist({'schema_id': 'test_schema/v0.1.0', 'data': [1, 2, 3]})

    # matching schema returns payload
    loaded = ctrl.fetch()
    assert loaded is not None
    assert loaded['data'] == [1, 2, 3]

    # controller configured with different schema raises on load
    other_ctrl = ctrl_mod.Controller[dict](
        fpath,
        schema_id='other_schema/v0.2.0',
    )
    with pytest.raises(ctrl_mod.ArtifactError, match='Schema mismatch'):
        other_ctrl.fetch()

    # missing file raises by default, returns None when missing_ok=True
    missing_ctrl = ctrl_mod.Controller[dict](tmp_path / 'missing.json')
    with pytest.raises(ctrl_mod.ArtifactError, match='missing'):
        missing_ctrl.fetch()
    assert missing_ctrl.fetch(missing_ok=True) is None
