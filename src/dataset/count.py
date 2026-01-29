'''Label counting related utilities'''

# standard imports
import os
# third-party imports
import numpy
import tqdm
# local imports
import dataset
import utils

# --------------------class distribution of each label layer--------------------
def count_label_cls(
        blks_fpaths: str,
        results_fpath: str,
        logger: utils.Logger,
        *,
        overwrite: bool
    ) -> dict[str, list[int]]:
    '''Aggregate label class counting.'''

    # get a child logger
    logger=logger.get_child('stats')

    # check if already counted
    if os.path.exists(results_fpath) and not overwrite:
        logger.log('INFO', f'Gathering label counts from: {results_fpath}')
        return utils.load_json(results_fpath)

    # load blocks dict
    blks_fpaths_dict = utils.load_json(blks_fpaths)

    # aggregate pixel count in each block
    count_results = {}
    for fpath in tqdm.tqdm(blks_fpaths_dict.values()):
        blk = dataset.DataBlock().load(fpath)
        for layer, counts in blk.meta['label_count'].items():
            cls_count = numpy.asarray(counts)
            if layer in count_results:
                count_results[layer] += cls_count
            else:
                count_results[layer] = cls_count
            count_results[layer] = [int(x) for x in count_results[layer]]

    # save to file and return
    logger.log('INFO', f'Label class distributions save to {results_fpath}')
    utils.write_json(results_fpath, count_results)
    return count_results
