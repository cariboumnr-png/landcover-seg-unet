# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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

'''
Dummy dataset generator for mosaic utility testing and local execution.
'''

# standard imports
import dataclasses
import pathlib
# third-party imports
import numpy
import rasterio
import rasterio.crs
import rasterio.transform
# local imports
import landseg.testing.tiff as tiff


# ----- private dataclass
@dataclasses.dataclass(frozen=True)
class PreMosaickingDummyPaths:
    '''Container for dummy TIFF file paths.'''
    root: str | pathlib.Path

    @property
    def image_17n(self) -> pathlib.Path:
        return pathlib.Path(self.root) / 'utm_rasters' / 'image_17n.tif'

    @property
    def image_18n(self) -> pathlib.Path:
        return pathlib.Path(self.root) / 'utm_rasters' / 'image_18n.tif'

    @property
    def label_17n(self) -> pathlib.Path:
        return pathlib.Path(self.root) / 'utm_rasters' / 'label_17n.tif'

    @property
    def label_18n(self) -> pathlib.Path:
        return pathlib.Path(self.root) / 'utm_rasters' / 'label_18n.tif'


# ----- public function
def generate_pre_mosaic_data(input_root: str | pathlib.Path) -> None:
    '''Generate the cross-UTM rasters under input root.

    Args:
        input_root:
            Root directory for output files.
    '''
    print('Generating cross-UTM rasters for mosaic utility testing...')
    pathlib.Path(input_root).mkdir(parents=True, exist_ok=True)
    paths = PreMosaickingDummyPaths(input_root)

    raw_shape = (100, 100)
    crs_17 = rasterio.crs.CRS.from_string('EPSG:32617')
    crs_18 = rasterio.crs.CRS.from_string('EPSG:32618')
    transform_17 = rasterio.transform.from_origin(500000.0, 5200000.0, 20.0, 20.0)
    transform_18 = rasterio.transform.from_origin(300000.0, 5200000.0, 20.0, 20.0)

    print(f'Creating raw feature in UTM17: {paths.image_17n}')
    tiff.create_dummy_geotiff(
        paths.image_17n,
        config=tiff.TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=crs_17,
            transform=transform_17,
            dtype=numpy.uint16,
            nodata=65535,
        ),
        data_gen_func=lambda s, _: tiff.gen_image_data(s, 1),
    )

    print(f'Creating raw feature in UTM18: {paths.image_18n}')
    tiff.create_dummy_geotiff(
        paths.image_18n,
        config=tiff.TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=crs_18,
            transform=transform_18,
            dtype=numpy.uint16,
            nodata=65535,
        ),
        data_gen_func=lambda s, _: tiff.gen_image_data(s, 1),
    )

    print(f'Creating raw label in UTM17: {paths.label_17n}')
    tiff.create_dummy_geotiff(
        paths.label_17n,
        config=tiff.TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=crs_17,
            transform=transform_17,
            dtype=numpy.uint8,
            nodata=255,
        ),
        data_gen_func=tiff.gen_label_data,
    )

    print(f'Creating raw label in UTM18: {paths.label_18n}')
    tiff.create_dummy_geotiff(
        paths.label_18n,
        config=tiff.TIFFConfig(
            shape=raw_shape,
            bands=1,
            crs=crs_18,
            transform=transform_18,
            dtype=numpy.uint8,
            nodata=255,
        ),
        data_gen_func=tiff.gen_label_data,
    )
