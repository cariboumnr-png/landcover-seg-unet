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

'''Label-count aggregation utilities.'''

# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core


# ----- `count_label` function
def count_label(block_file_list: list[str]) -> dict[str, list[int]]:
    '''Aggregate label class counts across a list of block files.'''
    # iterate current training blocks to get label class counts
    lbl_stats: dict[str, list[int]] = {}
    for fpath in block_file_list:
        blk_meta = geo_core.DataBlock.load(fpath).manifest
        for channel, counts in blk_meta['label_count'].items():
            cls_count = numpy.asarray(counts)
            if channel in lbl_stats:
                lbl_stats[channel] += cls_count
            else:
                lbl_stats[channel] = list(cls_count)
            lbl_stats[channel] = [int(x) for x in lbl_stats[channel]]
    return lbl_stats
