# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''Shared configuration and fixtures for the landseg test suite.'''

# standard imports
import pathlib
import pytest
# third-party imports
import numpy
import rasterio.transform
# local imports
import landseg.testing as testing

# absolute path to the repo root folder
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture
def dummy_data_paths():
    '''
    Fixture providing paths to the pre-generated dummy data.

    The created dummy data (.TIFF and .JSON) file paths match those that
    are defined in `root/configs/user.yaml` and are locally persisted.
    If any files do not exist, they will be automatically generated to
    ensure the integration tests pass out-of-the-box.
    '''
    in_dir = ROOT_DIR / 'experiment' / 'input'
    paths = testing.PreHarmonizationDummyPaths(root=in_dir)

    if not paths.all_paths_exist:
        testing.generate_pre_harmonization_data(in_dir)
    return paths


@pytest.fixture
def dummy_geotiff_factory(tmp_path):
    '''
    Factory fixture to create temporary, tiny GeoTIFF files for tests.

    Returns a function that creates a GeoTIFF and returns its file path.
    '''
    def _create_dummy_geotiff( # pylint: disable=too-many-arguments
        *,
        filename='dummy.tif',
        width=16,
        height=16,
        bands=3,
        crs='EPSG:3161',
        transform=None,
        dtype=numpy.uint8,
        nodata=0,
        data_gen_func=None
    ):
        file_path = tmp_path / filename
        tf = transform or rasterio.transform.from_origin(
            500000.0, 6000000.0, 20.0, 20.0
        )

        config = testing.TIFFConfig(
            shape=(height, width),
            bands=bands,
            crs=crs,
            transform=tf,
            dtype=dtype,
            nodata=nodata
        )


        def _default_gen(shape, _):
            return numpy.random.randint(0, 256, shape).astype(dtype)

        gen = data_gen_func or _default_gen

        testing.create_dummy_geotiff(
            str(file_path),
            config=config,
            data_gen_func=gen
        )

        return file_path

    return _create_dummy_geotiff
