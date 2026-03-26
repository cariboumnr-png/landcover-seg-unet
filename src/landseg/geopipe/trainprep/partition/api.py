# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Dataset partioning pipeline.
'''

# standard imports
import dataclasses
# third-party imports
import numpy
# local imports
import landseg.geopipe.trainprep.partition as partition
import landseg.utils as utils

# ------------------------------Public  Dataclass------------------------------
@dataclasses.dataclass
class PartitionConfig:
    '''Partition pipeline configuration.'''
    val_test_ratios: tuple[float, float]
    buffer_step: int
    reward_ratios: dict[int, float]     # 0-based
    scoring_alpha: float                # exponent for transforming block counts
    scoring_beta: float                 # reward weight for classes during L1
    max_skew_rate: float
    block_spec: tuple[int, int]         # block_size, block_stride

@dataclasses.dataclass
class BlockPartitions:
    '''Container for partioning results.'''
    train: list[tuple[int, int]]
    val: list[tuple[int, int]]
    test: list[tuple[int, int]]

# -------------------------------Public Function-------------------------------
def partition_blocks(
    base_class_counts: dict[tuple[int, int], list[int]],
    catalog_class_counts: dict[tuple[int, int], list[int]],
    config: PartitionConfig,
    logger: utils.Logger,
    *,
    block_spec: tuple[int, int]
) -> BlockPartitions:
    '''doc'''

    # split dataset
    base_count = numpy.array(list(base_class_counts.values()))
    splits = partition.stratified_splitter(
        base_count,
        val_ratio=config.val_test_ratios[0],
        test_ratio=config.val_test_ratios[1],
    )

    # filter candidate files (remove overlaps from val+test blocks)
    catalog_coords = list(catalog_class_counts.keys())
    exclude_coords = [catalog_coords[i] for i in [*splits.val, *splits.test]]
    safe_candidates = partition.filter_safe_tiles(
        catalog_coords,
        exclude_coords,
        block_size=block_spec[0],
        block_stride=block_spec[1],
        buffer_steps=config.buffer_step
    )

    # score and rank the safe candidate tiles
    global_class_count = numpy.sum(base_count, axis=0)
    blocks_to_score = {k: v for k, v in catalog_class_counts.items() if k in safe_candidates}
    ranked_candidates = partition.score_blocks(
        global_class_count,
        blocks_to_score,
        reward=tuple(config.reward_ratios.keys()),
        alpha=config.scoring_alpha,
        beta=config.scoring_beta
    )

    # hydrate
    train_class_count = list(numpy.sum(base_count[list(splits.train)], axis=0))
    selected, current_count = partition.hydrate_train_split(
        train_class_count,
        ranked_candidates,
        target_ratios=config.reward_ratios,
        max_skew_rate=config.max_skew_rate
    )

    # report
    logger.log('INFO', f'Hydration complete with {len(selected)} additional blocks')
    print([int(x) for x in current_count])
    print(numpy.array(current_count) / train_class_count)

    # return block coordinates for each split (train/val/test)
    return BlockPartitions(
        train=selected,
        val=[catalog_coords[i] for i in splits.val],
        test =[catalog_coords[i] for i in splits.test]
    )
