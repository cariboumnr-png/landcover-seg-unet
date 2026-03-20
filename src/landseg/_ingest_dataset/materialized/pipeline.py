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
Data preparation pipeline that maps rasters to a grid, builds and
normalizes blocks, splits datasets, and persists a versioned schema.

Public APIs:
    - build_catalogue_test: Run the end-to-end dataset workflow.
'''

# standard imports
# third-part
import numpy
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg._ingest_dataset.canonical as canonical
import landseg._ingest_dataset.materialized as materialized
import landseg.utils as utils

def materialize_dataset_test(
    base_grid: core.GridLayoutLike,
    config: configs.RootConfig,
    logger: utils.Logger,
):
    '''doc'''

    root = './experiment/artifacts/data_cache/branch_test'

    catalog = canonical.BlocksCatalog.from_json(f'{root}/fit/catalog.json')

    base_coords = list(base_grid.keys())
    base_catalog_entries = [
        v for v in catalog.values() if tuple(v['loc_col_row']) in base_coords
    ]

    counts = numpy.array([e['class_count'] for e in base_catalog_entries])
    splitted = materialized.stratified_splitter(counts, 0.10, 0.10)

    train_indices = [base_coords[i] for i in splitted.train]
    val_indices = [base_coords[i] for i in splitted.val]
    test_indices = [base_coords[i] for i in splitted.test]

    print(len(train_indices), len(val_indices), len(test_indices))
