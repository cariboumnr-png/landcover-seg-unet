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
Utilities for aggregating label class counts across cached blocks. Sums
per-block counts from serialized DataBlock artifacts to produce dataset-
level label distributions.

Public APIs:
    - count_label_class: Aggregate label counts across block artifacts
      and persist the combined distribution.
'''

# standard imports
import os
# third-party imports
import numpy
import tqdm
# local imports
import landseg.prep_raster.blockbuilder as blockbuilder
import landseg.utils as utils

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
        recount: If `False` and `results_fpath` exists, previously
            computed results are loaded and returned.

    Returns:
        dict: Mapping from label layer name to aggregated class counts.
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
        blk = blockbuilder.DataBlock().load(fpath)
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
    utils.hash_artifacts(output_fpath)
    return count_results
