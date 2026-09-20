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
Unit tests for `landseg.artifacts.payload_io`.
'''

# standard imports
import os
import typing
# third-party imports
import pytest
# local imports
import landseg.artifacts.controller as ctrl_mod
import landseg.artifacts.payload_io as payload_mod
import landseg.artifacts.policy as policy_mod


# ----- `PayloadController` tests
def test_payload_controller_save_and_load(tmp_path):
    '''
    Given: A split-file data path and structured payload dict.
    When: Calling `PayloadController.save()` and `PayloadController.load()`.
    Then: Write data file and metadata sidecar, restoring exact payload.
    '''
    data_path = str(tmp_path / 'catalog.json')
    meta_path = str(tmp_path / 'catalog_meta.json')

    ctrl = payload_mod.PayloadController[dict, dict](
        data_fpath=data_path,
        schema_id='v1_catalog',
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )

    payload_in: payload_mod.PayloadDict[dict, dict] = {
        'schema_id': 'v1_catalog',
        'artifact_meta': {'created_by': 'unit_test', 'version': 1},
        'data': {'items': [10, 20, 30]},
    }

    ctrl.save(payload_in)

    assert os.path.exists(data_path)
    assert os.path.exists(meta_path)

    loaded = ctrl.load()
    assert loaded is not None
    assert loaded['schema_id'] == 'v1_catalog'
    assert loaded['artifact_meta'] == {'created_by': 'unit_test', 'version': 1}
    assert loaded['data'] == {'items': [10, 20, 30]}


def test_payload_controller_schema_mismatch(tmp_path):
    '''
    Given: A stored payload with schema ID `schema_v1`.
    When: `PayloadController.load()` runs expecting `schema_v2_expected`.
    Then: Raise an `ArtifactError` indicating schema ID mismatch.
    '''
    data_path = str(tmp_path / 'schema_test.json')

    ctrl_writer = payload_mod.PayloadController[dict, dict](
        data_fpath=data_path,
        schema_id='schema_v1',
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )
    ctrl_writer.save({
        'schema_id': 'schema_v1',
        'artifact_meta': {},
        'data': {'foo': 'bar'},
    })

    ctrl_reader = payload_mod.PayloadController[dict, dict](
        data_fpath=data_path,
        schema_id='schema_v2_expected',
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )
    with pytest.raises(ctrl_mod.ArtifactError, match='Mismatch schema'):
        ctrl_reader.load()


def test_payload_controller_missing_files(tmp_path):
    '''
    Given: A non-existent payload file path.
    When: `PayloadController.load()` is called with `BUILD_IF_MISSING`.
    Then: Return `None` to indicate missing payload files.
    '''
    data_path = str(tmp_path / 'non_existent.json')

    ctrl = payload_mod.PayloadController[dict, dict](
        data_fpath=data_path,
        schema_id='schema_v1',
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )

    assert ctrl.load() is None


def test_payload_controller_save_validation(tmp_path):
    '''
    Given: Non-dictionary or missing required keys in payload dictionary.
    When: Calling `PayloadController.save()`.
    Then: Raise a TypeError or ValueError validating payload structure.
    '''
    data_path = str(tmp_path / 'validation.json')
    ctrl = payload_mod.PayloadController[dict, dict](
        data_fpath=data_path,
        schema_id='schema_v1',
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )

    with pytest.raises(TypeError, match='payload must be a dict'):
        ctrl.save('invalid_type')  # type: ignore

    with pytest.raises(ValueError, match='Missing payload keys'):
        ctrl.save({'schema_id': 'schema_v1', 'data': {}})  # type: ignore


def test_payload_controller_load_or_fail_success(tmp_path):
    '''
    Given: Stored payload files on disk.
    When: `load()` is invoked via `load_or_fail` or `load_json_or_fail`.
    Then: Return the loaded `PayloadDict` successfully.
    '''
    data_path = str(tmp_path / 'item.json')
    ctrl = payload_mod.PayloadController[dict, dict](
        data_fpath=data_path,
        schema_id='v1_item',
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )
    ctrl.save({
        'schema_id': 'v1_item',
        'artifact_meta': {'name': 'test'},
        'data': {'count': 42},
    })

    # test `load_or_fail` factory
    ctrl_fail = payload_mod.PayloadController[dict, dict].load_or_fail(
        data_path,
        schema_id='v1_item',
    )
    assert isinstance(ctrl_fail, payload_mod.LoadOrFailPayloadController)
    loaded = ctrl_fail.load()
    assert loaded['schema_id'] == 'v1_item'
    assert loaded['data'] == {'count': 42}

    # test `load_json_or_fail` factory
    ctrl_json_fail = (
        payload_mod.PayloadController[dict, dict].load_json_or_fail(
            data_path,
            schema_id='v1_item',
        )
    )
    assert isinstance(
        ctrl_json_fail, payload_mod.LoadOrFailPayloadController
    )
    assert ctrl_json_fail.load() == loaded


def test_payload_controller_load_or_fail_missing(tmp_path):
    '''
    Given: Non-existent payload files.
    When: `load()` is invoked on a `LoadOrFailPayloadController`.
    Then: Raise `ArtifactError` signaling missing artifact files.
    '''
    data_path = str(tmp_path / 'missing.json')
    ctrl = payload_mod.LoadOrFailPayloadController[dict, dict](
        data_path,
        schema_id='v1_item',
    )
    with pytest.raises(ctrl_mod.ArtifactError, match='Error loading'):
        ctrl.load()


def test_payload_controller_overload_signatures():
    '''
    Given: The `PayloadController.load` method.
    When: Introspecting registered typing overloads.
    Then: Register overloads for specialized and general return types.
    '''
    overloads = typing.get_overloads(payload_mod.PayloadController.load)
    assert len(overloads) == 2
