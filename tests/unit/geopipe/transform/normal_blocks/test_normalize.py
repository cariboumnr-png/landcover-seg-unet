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

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

'''Unit tests for block normalization logic (normalize.py).'''

# standard imports
import os
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.transform.normal_blocks.normalize as normalize


# ----- `_normalize_image` tests
def test_normalize_image_math():
    # [1, 2, 2]
    raw_img = numpy.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=numpy.float32)
    # pixel (1,1) is invalid
    mask = numpy.array([[True, True], [True, False]], dtype=bool)

    _stats: geo_core.ImageBandStats = {
        'total_count': 100,
        'current_mean': 20.0,
        'accum_m2': 100.0,
        'std': 5.0
    }
    global_stats = {'band_0': _stats}

    norm_img = normalize._normalize_image(raw_img, mask, global_stats)

    # pixel 0,0: (10.0 - 20.0)/5.0 = -2.0
    # pixel 0,1: (20.0 - 20.0)/5.0 = 0.0
    # pixel 1,0: (30.0 - 20.0)/5.0 = 2.0
    # pixel 1,1: invalid, replaced by mean (20.0), so (20.0 - 20.0)/5.0 = 0.0
    expected = numpy.array([[[-2.0, 0.0], [2.0, 0.0]]], dtype=numpy.float32)
    assert numpy.allclose(norm_img, expected)


# ----- `_purge` tests
def test_purge_removes_stale_files(tmp_path):
    target_dir = tmp_path / 'normalized_blocks'
    os.makedirs(target_dir, exist_ok=True)

    # create expected and stale files
    expected_file = target_dir / 'block_1.npz'
    stale_file = target_dir / 'block_stale.npz'

    expected_file.touch()
    stale_file.touch()

    filenames_to_keep = ['block_1.npz']

    purged_count = normalize._purge(filenames_to_keep, str(target_dir))

    # should remove block_stale.npz and return 1
    assert purged_count == 1
    assert os.path.exists(expected_file)
    assert not os.path.exists(stale_file)
