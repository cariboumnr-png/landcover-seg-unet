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
import pytest
# third-party imports
import numpy
import rasterio
# local imports
import scripts.generate_dummy_data as generate_dummy_data

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

        config = generate_dummy_data.TIFFConfig(
            shape=(height, width),
            bands=bands,
            crs='+proj=latlong',
            transform=rasterio.transform.from_origin(0, 0, 1, 1),
            dtype=dtype
        )

        def _data_gen_func(shape, _):
            return numpy.random.randint(0, 256, shape).astype(dtype)

        generate_dummy_data.create_dummy_geotiff(
            str(file_path),
            config=config,
            data_gen_func=_data_gen_func
        )

        return file_path

    return _create_dummy_geotiff
