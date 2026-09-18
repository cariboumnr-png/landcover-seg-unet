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

'''Unit tests for prepared schema builder logic (schema.py).'''

# local imports
import landseg.geopipe.prepare.materialize_blocks.schema as schema


# ----- `build_schema` tests
def test_build_schema(mocker):
    '''
    Given: Mocked controller values for prepared blocks, stats,
        and paths.
    When: Running build_schema.
    Then: Correctly construct the dataset schema, compile checksums,
        and log the schema report.
    '''
    # mock SchemaCtrl
    mock_schema_ctrl = mocker.Mock()
    mock_schema_ctrl.fetch.return_value = None # force schema creation
    mocker.patch(
        'landseg.geopipe.prepare.schema.SchemaCtrl',
        return_value=mock_schema_ctrl
    )

    # mock Controller.load_json_or_fail
    def mock_load(filepath):
        mock_ctrl = mocker.Mock()
        mock_ctrl.sha256 = 'mock-hash-value'
        if (
            'splits_prepared_blocks' in filepath or
            'block_splits' in filepath or
            'prepared' in filepath
        ):
            mock_ctrl.fetch.return_value = {
                'train': {'block_0': 'path/to/block_0.npz'},
                'val': {},
                'test': {}
            }
        elif 'label_stats' in filepath:
            mock_ctrl.fetch.return_value = {
                'head1': [100, 200]
            }
        elif 'image_stats' in filepath:
            mock_ctrl.fetch.return_value = {
                'band_0': {'mean': 1.0}
            }
        else:
            mock_ctrl.fetch.return_value = {}
        return mock_ctrl

    mocker.patch(
        'landseg.artifacts.Controller.load_json_or_fail',
        side_effect=mock_load
    )

    # mock logger
    mock_logger = mocker.Mock()

    # paths mock
    mock_paths = mocker.Mock()
    mock_paths.schema = 'schema.json'
    mock_paths.splits_source_blocks = 'block_source.json'
    mock_paths.splits_prepared_blocks = 'block_splits.json'
    mock_paths.label_stats = 'label_stats.json'
    mock_paths.image_stats = 'image_stats.json'

    # mock context
    mock_context = mocker.Mock()
    mock_context.targets.head_names = ['head1', 'head1_sub']
    mock_context.targets.head_parent = {
        'head1': None,
        'head1_sub': 'head1',
    }
    mock_context.targets.head_parent_cls = {
        'head1': None,
        'head1_sub': 1,
    }
    mock_context.targets.num_classes = {
        'head1': 2,
        'head1_sub': 1,
    }
    mock_context.targets.class_names = {
        'head1': ['cls0', 'cls1'],
        'head1_sub': ['sub0'],
    }
    mock_context.targets.ignore_classes = {
        'head1': [255],
        'head1_sub': [255],
    }

    schema.build_schema(
        mock_paths,
        mock_context,
        policy=mocker.Mock(),
        logger=mock_logger
    )

    # verify schema creation calls
    assert mock_schema_ctrl.persist.called
    persisted_schema = mock_schema_ctrl.persist.call_args[0][0]
    assert persisted_schema['schema_version'] is not None
    assert persisted_schema['checksums']['block_source'] == 'mock-hash-value'
    assert (
        persisted_schema['checksums']['block_prepared'] == 'mock-hash-value'
    )
    assert persisted_schema['heads']['head_names'] == ['head1', 'head1_sub']
    assert persisted_schema['heads']['head_parent'] == {
        'head1': None,
        'head1_sub': 'head1',
    }
    assert persisted_schema['heads']['head_parent_cls'] == {
        'head1': None,
        'head1_sub': 1,
    }
    assert persisted_schema['heads']['num_classes'] == {
        'head1': 2,
        'head1_sub': 1,
    }
    assert mock_logger.set_schema_report.called
