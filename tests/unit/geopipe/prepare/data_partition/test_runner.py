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

'''Unit tests for dataset partition runner.'''

# standard imports
import dataclasses
# third-party imports
import rasterio.transform
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.prepare.common as common
import landseg.geopipe.prepare.data_context as data_context
import landseg.geopipe.prepare.data_partition.runner as runner
import landseg.geopipe.prepare.data_partition.operations as operations


# ----- test helper classes
@dataclasses.dataclass
class _DummyPaths:
    '''Dummy pipeline paths container.'''
    splits_source_blocks: str
    splits_summary: str
    label_stats: str


# ----- `run_datablocks_partition` tests
def test_run_datablocks_partition(tmp_path, mocker):
    '''
    Given: A DatasetContext with in-memory class counts.
    When: Running run_datablocks_partition.
    Then: Create splits, save summary with focal head, and persist
        paths.
    '''
    paths = _DummyPaths(
        splits_source_blocks=str(tmp_path / 'block_source.json'),
        splits_summary=str(tmp_path / 'summary.json'),
        label_stats=str(tmp_path / 'label_stats.json'),
    )

    catalog_view = data_context.DataBlocksView(
        valid_blocks={(0, 0): 'path/to/block_0.npz'},
        external_test_blocks=None,
        crs='EPSG:3161',
        transform=rasterio.transform.Affine.identity(),
        valid_class_counts={(0, 0): [0, 100]},
        base_class_counts={(0, 0): [0, 100]},
        focal_head='landcover_group',
    )
    features = data_context.FeatureSelection(names=('blue',), indices=(0,))
    targets = data_context.TargetHeadsContext(
        head_names=['landcover_group'],
        head_parent={'landcover_group': None},
        head_parent_cls={'landcover_group': None},
        num_classes={'landcover_group': 2},
        class_names={'landcover_group': ['VEG', 'WAT']},
        ignore_classes={'landcover_group': [255]},
        resolved_reclass={'landcover': None},
    )
    ctx = data_context.DatasetContext(
        catalog=catalog_view,
        features=features,
        targets=targets,
    )

    partition_config = operations.PartitionParameters(
        val_test_ratios=(0.0, 0.0),
        buffer_step=1,
        reward_ratios={},
        scoring_alpha=1.0,
        scoring_beta=0.0,
        max_skew_rate=1.0,
        block_spec=(256, 256, 128, 128),
    )

    logger = common.PreparationLogger(
        name='test_prep',
        log_file=str(tmp_path / 'report.txt'),
        enable_file_log=False,
    )
    logger.init_summary(run_id='test')

    runner.run_datablocks_partition(
        ctx,
        paths,
        partition_config,
        policy=artifacts.LifecyclePolicy.REBUILD,
        logger=logger,
    )

    # verify summary was saved with the focal head from context
    summary_ctrl = artifacts.Controller[dict].load_json_or_fail(
        paths.splits_summary
    )
    summary = summary_ctrl.fetch()
    assert summary['hydration']['focal_head'] == 'landcover_group'

    # verify splits source blocks saved
    src_ctrl = artifacts.Controller[dict].load_json_or_fail(
        paths.splits_source_blocks
    )
    src = src_ctrl.fetch()
    assert 'block_0' in src['train']
    assert src['train']['block_0'] == 'path/to/block_0.npz'
