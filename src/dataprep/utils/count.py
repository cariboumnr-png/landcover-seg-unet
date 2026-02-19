'''
Utilities for aggregating label class counts across cached blocks.

This module provides helpers to compute dataprep-level label
distributions by summing per-block class counts stored in
serialized `DataBlock` artifacts.
'''

# standard imports
import os
# third-party imports
import numpy
import tqdm
# local imports
import dataprep
import utils

# -------------------------------Public Function-------------------------------
def count_label_class(
    input_blocks: str,
    output_fpath: str,
    logger: utils.Logger,
    *,
    recount: bool =  False
) -> dict[str, list[int]]:
    '''
    Aggregate label class counts across a set of data blocks.

    Args:
        input_blocks: Path to a JSON file mapping block names to `.npz`
            file paths.
        output_fpath: Path to the output JSON file for aggregated class
            counts.
        logger: Logger used for progress and status reporting.
        overwrite: If `False` and `results_fpath` exists, previously
            computed results are loaded and returned.

    Returns:
        dict: Mapping from label layer name to aggregated class counts.

    --------------------------------------------------------------------
    Notes: This function loads each `DataBlock` and sums `label_count`
    metadata across all blocks. It is typically used for computing
    dataprep-wide label distributions for training diagnostics.
    '''

    # get a child logger
    logger=logger.get_child('stats')

    # check if already counted
    if os.path.exists(output_fpath) and not recount:
        logger.log('INFO', f'Gathering label counts from: {output_fpath}')
        return utils.load_json(output_fpath)

    # load blocks dict
    blks_fpaths_dict = utils.load_json(input_blocks)

    # aggregate pixel count in each block
    count_results = {}
    for fpath in tqdm.tqdm(blks_fpaths_dict.values()):
        blk = dataprep.DataBlock().load(fpath)
        for layer, counts in blk.meta['label_count'].items():
            cls_count = numpy.asarray(counts)
            if layer in count_results:
                count_results[layer] += cls_count
            else:
                count_results[layer] = cls_count
            count_results[layer] = [int(x) for x in count_results[layer]]

    # save to file and return
    logger.log('INFO', f'Label class distributions save to {output_fpath}')
    utils.write_json(output_fpath, count_results)
    return count_results
