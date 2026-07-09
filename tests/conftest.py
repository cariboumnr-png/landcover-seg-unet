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
Shared configuration and fixtures for the landseg test suite.
'''

# standard imports
import os
import pytest
# third-party imports
import numpy
import rasterio.transform
# local imports
import landseg.testing as testing

# Absolute path to the repo root folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


@pytest.fixture
def dummy_data_paths():
    '''Fixture providing paths to the pre-generated dummy data.'''
    ref_fpath = os.path.join(
        ROOT_DIR, 'experiment', 'input', 'extent_reference',
        'example_extent.tif'
    )
    dev_image = os.path.join(
        ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'dev',
        'example_image.tif'
    )
    dev_label = os.path.join(
        ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'dev',
        'example_label.tif'
    )
    test_image = os.path.join(
        ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'test',
        'example_image.tif'
    )
    test_label = os.path.join(
        ROOT_DIR, 'experiment', 'input', 'data', 'demo_data', 'test',
        'example_label.tif'
    )
    dataset_config = os.path.join(
        ROOT_DIR, 'experiment', 'input', 'data', 'demo_data',
        'example_config.json'
    )

    return {
        'ref_fpath': ref_fpath,
        'dev_image': dev_image,
        'dev_label': dev_label,
        'test_image': test_image,
        'test_label': test_label,
        'dataset_config': dataset_config
    }


@pytest.fixture
def dummy_geotiff_factory(tmp_path):
    '''
    Factory fixture to create temporary, tiny GeoTIFF files for tests.

    Returns a function that creates a GeoTIFF and returns its file path.
    '''

    def _create_dummy_geotiff(
        filename='dummy.tif',
        width=16,
        height=16,
        bands=3,
        dtype=numpy.uint8
    ):
        file_path = tmp_path / filename

        config = testing.TIFFConfig(
            shape=(height, width),
            bands=bands,
            crs='+proj=latlong',
            transform=rasterio.transform.from_origin(0, 0, 1, 1),
            dtype=dtype
        )

        def _data_gen_func(shape, _):
            return numpy.random.randint(0, 256, shape).astype(dtype)

        testing.create_dummy_geotiff(
            str(file_path),
            config=config,
            data_gen_func=_data_gen_func
        )

        return file_path

    return _create_dummy_geotiff
