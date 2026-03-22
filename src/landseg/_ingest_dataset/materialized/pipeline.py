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

    logger.log('INFO', 'Test')
    root = './experiment/artifacts/data_cache/branch_test'

    # from catalog.json and the base grid
    catalog = canonical.BlocksCatalog.from_json(f'{root}/fit/catalog.json')
    catalog = {k: v for k, v in catalog.items() if v['valid_px']}
    catalog_keys = list(catalog.keys())
    base_coords = list(base_grid.keys())
    base_catalog_entries = [
        v for v in catalog.values() if tuple(v['loc_col_row']) in base_coords
    ]
    counts = numpy.array([e['class_count'] for e in base_catalog_entries])

    # splitting
    splitted = materialized.stratified_splitter(counts, 0.10, 0.10)
    train_indices = [tuple(catalog[catalog_keys[i]]['loc_col_row']) for i in splitted.train]
    val_indices = [tuple(catalog[catalog_keys[i]]['loc_col_row']) for i in splitted.val]
    test_indices = [tuple(catalog[catalog_keys[i]]['loc_col_row']) for i in splitted.test]
    print(len(train_indices), len(val_indices), len(test_indices))
    excluded_indices = [(x[0], x[1]) for x in val_indices + test_indices]
    train_class_count = numpy.sum(counts[list(splitted.train)], axis=0)

    # train blocks hydration
    # filter overlapping tiles
    catalouged_coords = [(v['loc_col_row'][0], v['loc_col_row'][1]) for v in catalog.values()]
    safe_train_indices = materialized.filter_safe_tiles(
        catalouged_coords,
        excluded_indices,
        block_size=base_grid.tile_size[0],
        stride=128,
        buffer_steps=1
    )
    print(len(safe_train_indices))

    # scoring
    global_class_count = numpy.sum(counts, axis=0)
    input_blocks = {
        v['block_name']: v['class_count'] for v in catalog.values()
        if tuple(v['loc_col_row']) in safe_train_indices
    }
    materialized.score_blocks(
        global_class_count,
        input_blocks,
        materialized.ScoringConfig(
            alpha=1.0,
            beta=config.prep.data.scoring.beta,
            epsilon=config.prep.data.scoring.epsilon,
            reward=(2, 4)
        ),
        f'{root}/fit/scores.json'
    )

    # hydrate
    print(global_class_count)
    print(train_class_count)
    scores = utils.load_json(f'{root}/fit/scores.json')
    selected, current_count = materialized.hydrate_train_split(
        list(train_class_count),
        [(k, v['count']) for k, v in scores.items()],
        {2: 5.0, 4: 5.0}
    )
    print(len(selected))
    print([int(x) for x in current_count])

    print(numpy.array(current_count) / train_class_count)
