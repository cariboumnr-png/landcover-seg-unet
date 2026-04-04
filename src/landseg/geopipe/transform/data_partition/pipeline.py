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
Dataset partitioning pipeline.

Consumes a canonical blocks catalog and produces experiment-specific
dataset splits (train/val/test) based on stratified sampling, spatial
buffering, and class-balance heuristics. Outputs split manifests and
label statistics for downstream normalization and schema generation.
'''

# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.data_partition as data_partition
import landseg.utils as utils

# -------------------------------Public Function-------------------------------
def run_datablocks_partition(
    foundation_root: str,
    transform_root: str,
    partition_config: data_partition.PartitionParameters,
    logger: utils.Logger,
) -> None:
    '''
    Partition canonical data blocks into train/val/test splits.

    Loads the foundation catalog, optionally incorporates external test
    (holdout) blocks, and performs stratified splitting followed by
    spatially safe hydration of training blocks. Writes split manifests
    and aggregated label statistics to the transform directory.

    Artifacts written:
    - `block_source.json`: block file paths indexed by split
    - `label_stats.json`: aggregated label counts over training blocks

    Args:
        foundation_root: Root directory containing canonical catalog
            artifacts.
        transform_root: Output directory for transform-stage artifacts.
        partition_config: Parameters controlling split and balancing
            behavior.
        logger: Logger for progress and diagnostic output.
    '''

    # block size (row, col)
    blk_size = (partition_config.block_spec[0], partition_config.block_spec[1])

    # locate catalog JSON files
    dev_catalog = f'{foundation_root}/model_dev/catalog.json'
    test_catalog = f'{foundation_root}/test_holdout/catalog.json'

    # try load test catalog
    ext_test_blks: list[str] = []
    try:
        test = data_partition.parse_catalog(test_catalog, blk_size)
        logger.log('INFO', 'Use external test blocks, no test blocks split')
        ext_test_blks = list(test.valid_file_paths.values())
    except FileNotFoundError:
        logger.log('INFO', 'No external test blocks provided')
        if not partition_config.val_test_ratios[1]:
            logger.log('WARNING', 'No test blocks is to be included')

    # load model dev catalog
    dev = data_partition.parse_catalog(dev_catalog, blk_size)
    if not dev.base_class_counts:
        raise ValueError('No valid data catalog')

    # get split blocks
    logger.log('INFO', 'Split data blocks')
    blks_src, summary = data_partition.create_blocks_partition(
        dev.base_class_counts,
        dev.valid_class_counts,
        dev.valid_file_paths,
        partition_config,
        ext_test_blks=ext_test_blks
    )
    # log
    logger.log('INFO', f'Training blocks: {len(blks_src['train'])} ')
    for m in summary:
        logger.log('INFO', m)

    # iterate current traing blocks to get label class counts
    lbl_stats = _count_label(list(blks_src['train'].values()))

    # write JSON artifacts
    utils.write_json(f'{transform_root}/block_source.json', blks_src)
    utils.hash_artifacts(f'{transform_root}/block_source.json')

    utils.write_json(f'{transform_root}/label_stats.json', lbl_stats)
    utils.hash_artifacts(f'{transform_root}/label_stats.json')

def _count_label(block_file_list: list[str]) -> dict[str, list[int]]:
    '''Aggregate label class counts across a list of block files.'''

    # iterate current traing blocks to get label class counts
    lbl_stats: dict[str, list[int]] = {}
    for fpath in block_file_list:
        blk_meta = geo_core.DataBlock.load(fpath).meta
        for channel, counts in blk_meta['label_count'].items():
            cls_count = numpy.asarray(counts)
            if channel in lbl_stats:
                lbl_stats[channel] += cls_count
            else:
                lbl_stats[channel] = list(cls_count)
            lbl_stats[channel] = [int(x) for x in lbl_stats[channel]]
    return lbl_stats
