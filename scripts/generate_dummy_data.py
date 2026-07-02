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
Utility script to generate dummy/mock geospatial datasets (GeoTIFFs)
and corresponding configurations for local pipeline runs and testing.
'''

# standard imports
import json
import os
import sys
import typing
# third-party imports
import numpy
import rasterio
import rasterio.transform

# --------------------------------Public Function-------------------------------
def create_dummy_geotiff(
    fpath: str,
    *,
    shape: tuple[int, int],
    bands: int,
    crs: str,
    transform: rasterio.transform.Affine,
    dtype: typing.Any,
    data_gen_func: typing.Callable[[tuple[int, int], int], numpy.ndarray],
) -> None:
    '''Write a multi-band GeoTIFF with coordinate metadata.

    Args:
        fpath: Output file path.
        shape: Height and width in pixels.
        bands: Number of bands to write.
        crs: Coordinate Reference System string.
        transform: Affine geospatial transform.
        dtype: NumPy datatype of pixels.
        data_gen_func: Callback generating data per band.
    '''

    # make sure output directory exists
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    # set nodata based on dtype
    nodata_val = (
        0 if dtype == numpy.uint8
        else 255 if dtype == numpy.uint16
        else -9999.0
    )
    # write GeoTIFF
    with rasterio.open(
        fpath,
        'w',
        driver='GTiff',
        height=shape[0],
        width=shape[1],
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_val,
    ) as dst:
        for b in range(1, bands + 1):
            band_data = data_gen_func(shape, b)
            dst.write(band_data, b)

def generate_dummy_data(input_root: str = './experiment/input') -> None:
    '''Generate the full dummy dataset under input root.

    Args:
        input_root: Root directory for output files.
    '''

    print('Generating dummy geospatial data for landseg pipeline...')
    # spatial parameters matching configs/user.yaml
    crs = 'EPSG:2958'
    px_size = 10.0
    orig_x = 500000.0
    orig_y = 5000000.0
    width, height = 512, 512 # this gives 4 256*256 tiles per image
    shape = (height, width)
    
    # Combined extent shape covers both dev and test side-by-side
    extent_shape = (height, width * 2)

    # create affine transforms for spatial referencing
    dev_transform = rasterio.transform.from_origin(
        orig_x, orig_y, px_size, px_size
    )
    # offset test image by 512 pixels * 10m/px = 5120 meters
    test_transform = rasterio.transform.from_origin(
        orig_x + (width * px_size), orig_y, px_size, px_size
    )
    extent_transform = dev_transform

    # create extent reference (single band constant value on the wide extent)
    extent_path = f'{input_root}/extent_reference/example_extent.tif'
    print(f'Creating extent reference: {extent_path}')
    create_dummy_geotiff(
        extent_path,
        shape=extent_shape,
        bands=1,
        crs=crs,
        transform=extent_transform,
        dtype=numpy.uint8,
        data_gen_func=lambda s, b: numpy.ones(s, dtype=numpy.uint8),
    )

    # dev and test images w/ 4 bands: DEM+RGB
    dev_image_path = f'{input_root}/data/demo_data/dev/example_image.tif'
    test_image_path = f'{input_root}/data/demo_data/test/example_image.tif'

    print(f'Creating dev image: {dev_image_path}')
    create_dummy_geotiff(
        dev_image_path,
        shape=shape,
        bands=4,
        crs=crs,
        transform=dev_transform,
        dtype=numpy.float32,  # float32 due to DEM float band
        data_gen_func=_gen_image_data,
    )

    print(f'Creating test image: {test_image_path}')
    create_dummy_geotiff(
        test_image_path,
        shape=shape,
        bands=4,
        crs=crs,
        transform=test_transform,
        dtype=numpy.float32,
        data_gen_func=_gen_image_data,
    )

    # dev and test labels
    dev_label_path = f'{input_root}/data/demo_data/dev/example_label.tif'
    test_label_path = f'{input_root}/data/demo_data/test/example_label.tif'

    print(f'Creating dev label: {dev_label_path}')
    create_dummy_geotiff(
        dev_label_path,
        shape=shape,
        bands=1,
        crs=crs,
        transform=dev_transform,
        dtype=numpy.uint8,
        data_gen_func=_gen_label_data,
    )

    print(f'Creating test label: {test_label_path}')
    create_dummy_geotiff(
        test_label_path,
        shape=shape,
        bands=1,
        crs=crs,
        transform=test_transform,
        dtype=numpy.uint8,
        data_gen_func=_gen_label_data,
    )

    # domain knowledge layers (covering the wide extent)
    domain_1_path = f'{input_root}/domain_knowledge/example_domain_1.tif'
    domain_2_path = f'{input_root}/domain_knowledge/example_domain_2.tif'

    print(f'Creating domain knowledge 1: {domain_1_path}')
    create_dummy_geotiff(
        domain_1_path,
        shape=extent_shape,
        bands=1,
        crs=crs,
        transform=extent_transform,
        dtype=numpy.uint8,
        data_gen_func=lambda s, b: numpy.random.randint(
            1, 5, size=s, dtype=numpy.uint8
        ),
    )

    print(f'Creating domain knowledge 2: {domain_2_path}')
    create_dummy_geotiff(
        domain_2_path,
        shape=extent_shape,
        bands=1,
        crs=crs,
        transform=extent_transform,
        dtype=numpy.uint8,
        data_gen_func=lambda s, b: numpy.random.randint(
            1, 10, size=s, dtype=numpy.uint8
        ),
    )

    # dataset config JSON
    config_path = f'{input_root}/data/demo_data/example_config.json'
    print(f'Creating dataset configuration: {config_path}')
    config_data = {
        'image_band_map': {
            'dem': 0,
            'red': 1,
            'green': 2,
            'blue': 3
        },
        'label_specs': {
            'two_classes': {
                'num_cls': 2,
                'ignore_cls': [255]
            }
        },
    }

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)

    print('\nDummy data generation completed successfully!')

# -------------------------------Private Functions-----------------------------
def _gen_image_data(shape: tuple[int, int], band_idx: int) -> numpy.ndarray:
    '''Generate dummy image band or terrain elevation data.'''

    # first band as DEM (1-based)
    if band_idx == 1:
        # generates a gradient representing terrain elevation
        x = numpy.linspace(100.0, 300.0, shape[1])
        return numpy.tile(x, (shape[0], 1)).astype(numpy.float32)
    # random mock values for visual image bands
    return numpy.random.randint(
        100, 1000, size=shape, dtype=numpy.uint16
    )

def _gen_label_data(shape: tuple[int, int], _: int) -> numpy.ndarray:
    '''Generate dummy label data with ignore index.'''

    labels = numpy.random.choice([0, 1, 255], size=shape, p=[0.45, 0.45, 0.10])
    return labels.astype(numpy.uint8)

# -------------------------------Main Executable-------------------------------
if __name__ == '__main__':
    
    # check if the directory already exists and is not empty
    target_dir = './experiment/input'
    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(
            f'WARNING: Target directory "{target_dir}" ' 
            f'already exists and is not empty.'
        )
        response = input(
            'Generating dummy data will overwrite existing files. Proceed? [y/N]: '
        )
        if response.strip().lower() not in ('y', 'yes'):
            print('Aborted.')
            sys.exit(0)
    generate_dummy_data(target_dir)