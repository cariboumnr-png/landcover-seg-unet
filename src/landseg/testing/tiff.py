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

'''
Dummy dataset generator for landseg integration testing and local execution.
'''

# standard imports
import dataclasses
import pathlib
import typing
# third-party imports
import numpy
import rasterio
import rasterio.transform


# ----- public dataclass
@dataclasses.dataclass
class TIFFConfig:
    '''Container for creating a dummy GeoTIFF file via `rasterio`.'''
    shape: tuple[int, int]
    bands: int
    crs: str
    transform: rasterio.transform.Affine
    dtype: typing.Any
    nodata: int | float


# ----- public function
def create_dummy_geotiff(
    fpath: str | pathlib.Path,
    *,
    config: TIFFConfig,
    data_gen_func: typing.Callable[[tuple[int, int], int], numpy.ndarray],
) -> None:
    '''Write a multi-band GeoTIFF with coordinate metadata.

    Args:
        fpath:
            Output file path.
        config:
            TIFFConfig with shape, bands, crs, transform, dtype, nodata.
        data_gen_func:
            Callback generating data per band.
    '''
    pathlib.Path(fpath).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        fpath,
        'w',
        driver='GTiff',
        height=config.shape[0],
        width=config.shape[1],
        count=config.bands,
        dtype=config.dtype,
        crs=config.crs,
        transform=config.transform,
        nodata=config.nodata,
    ) as dst:
        for b in range(1, config.bands + 1):
            band_data = data_gen_func(config.shape, b)
            dst.write(band_data, b)


def gen_image_data(shape: tuple[int, int], band_idx: int) -> numpy.ndarray:
    '''Generate dummy image band or terrain elevation data.'''
    if band_idx == 1:
        x = numpy.linspace(100.0, 300.0, shape[1])
        return numpy.tile(x, (shape[0], 1)).astype(numpy.float32)
    return numpy.random.randint(100, 1000, size=shape, dtype=numpy.uint16)


def gen_label_data(shape: tuple[int, int], _: int) -> numpy.ndarray:
    '''Generate dummy label data with ignore index.'''
    labels = numpy.random.choice([1, 2, 255], size=shape, p=[0.45, 0.45, 0.10])
    return labels.astype(numpy.uint8)
