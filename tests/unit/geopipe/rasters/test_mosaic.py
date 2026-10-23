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

'''Unit tests for multi-unit raster virtual reprojection and mosaicking.'''

# standard imports
import pathlib
import xml.etree.ElementTree
# third-party imports
import numpy
import pytest
import rasterio
import rasterio.transform
# local imports
import landseg.geopipe.rasters.mosaic as mosaic


# ----- `build_vrt_mosaic` input validation tests
def test_build_vrt_mosaic_empty_inputs_raises(tmp_path):
    '''
    Given: An empty input paths sequence.
    When: Invoking `build_vrt_mosaic`.
    Then: Raises ValueError indicating empty inputs.
    '''
    out_vrt = tmp_path / 'mosaic.vrt'
    with pytest.raises(ValueError, match='cannot be empty'):
        mosaic.build_vrt_mosaic(
            input_paths=[],
            output_path=out_vrt,
        )


def test_build_vrt_mosaic_missing_file_raises(tmp_path):
    '''
    Given: An input path to a non-existent raster file.
    When: Invoking `build_vrt_mosaic`.
    Then: Raises FileNotFoundError.
    '''
    out_vrt = tmp_path / 'mosaic.vrt'
    missing = tmp_path / 'nonexistent.tif'
    with pytest.raises(FileNotFoundError, match='Source raster not found'):
        mosaic.build_vrt_mosaic(
            input_paths=[missing],
            output_path=out_vrt,
        )


def test_build_vrt_mosaic_mismatched_band_count_raises(
    dummy_geotiff_factory,
    tmp_path,
):
    '''
    Given: Two rasters with differing band counts (10 bands vs 1 band).
    When: Invoking `build_vrt_mosaic`.
    Then: Raises ValueError indicating band count mismatch.
    '''
    t1 = dummy_geotiff_factory(
        filename='tile1_10band.tif',
        bands=10,
        crs='EPSG:3161',
    )
    t2 = dummy_geotiff_factory(
        filename='tile2_1band.tif',
        bands=1,
        crs='EPSG:3161',
    )
    out_vrt = tmp_path / 'mosaic.vrt'
    with pytest.raises(ValueError, match='Band count mismatch'):
        mosaic.build_vrt_mosaic(
            input_paths=[t1, t2],
            output_path=out_vrt,
        )


def test_build_vrt_mosaic_mismatched_dtype_raises(
    dummy_geotiff_factory,
    tmp_path,
):
    '''
    Given: Two rasters with incompatible data types (uint8 vs float32).
    When: Invoking `build_vrt_mosaic`.
    Then: Raises ValueError indicating data types mismatch.
    '''
    t1 = dummy_geotiff_factory(
        filename='tile1_u8.tif',
        bands=1,
        dtype=numpy.uint8,
        crs='EPSG:3161',
    )
    t2 = dummy_geotiff_factory(
        filename='tile2_f32.tif',
        bands=1,
        dtype=numpy.float32,
        crs='EPSG:3161',
    )
    out_vrt = tmp_path / 'mosaic.vrt'
    with pytest.raises(ValueError, match='Data types mismatch'):
        mosaic.build_vrt_mosaic(
            input_paths=[t1, t2],
            output_path=out_vrt,
        )


# ----- `build_vrt_mosaic` multi-unit reprojection tests
def test_build_vrt_mosaic_multi_utm_zones_to_epsg3161(
    dummy_geotiff_factory,
    tmp_path,
):
    '''
    Given: Rasters from two management units in UTM 16N and UTM 17N.
    When: Assembling a virtual mosaic reprojected to EPSG:3161.
    Then: Produces a valid multi-band VRT in EPSG:3161 readable by rasterio.
    '''
    tf1 = rasterio.transform.from_origin(300000.0, 5000000.0, 20.0, 20.0)
    t1 = dummy_geotiff_factory(
        filename='unit1_utm16.tif',
        width=16,
        height=16,
        bands=3,
        crs='EPSG:32616',
        transform=tf1,
        dtype=numpy.uint16,
        nodata=0,
    )

    tf2 = rasterio.transform.from_origin(500000.0, 5000000.0, 20.0, 20.0)
    t2 = dummy_geotiff_factory(
        filename='unit2_utm17.tif',
        width=16,
        height=16,
        bands=3,
        crs='EPSG:32617',
        transform=tf2,
        dtype=numpy.uint16,
        nodata=0,
    )

    out_vrt = tmp_path / 'multi_unit_mosaic.vrt'
    result_path = mosaic.build_vrt_mosaic(
        input_paths=[t1, t2],
        output_path=out_vrt,
        target_crs='EPSG:3161',
        target_resolution=20.0,
    )

    assert result_path == out_vrt
    assert out_vrt.exists()

    with rasterio.open(out_vrt) as ds:
        assert ds.crs.to_string() == 'EPSG:3161'
        assert ds.count == 3
        assert ds.dtypes[0] == 'uint16'
        assert ds.res == (20.0, 20.0)


def test_build_vrt_mosaic_same_crs_aligned(
    dummy_geotiff_factory,
    tmp_path,
):
    '''
    Given: Two adjacent rasters already aligned in EPSG:3161.
    When: Building a virtual mosaic.
    Then: Assembles them with correct total extent and dimensions.
    '''
    tf1 = rasterio.transform.from_origin(500000.0, 6000000.0, 20.0, 20.0)
    t1 = dummy_geotiff_factory(
        filename='tile_west.tif',
        width=20,
        height=20,
        bands=2,
        crs='EPSG:3161',
        transform=tf1,
        dtype=numpy.float32,
        nodata=-9999.0,
    )

    # 400m east (20 pixels * 20m)
    tf2 = rasterio.transform.from_origin(500400.0, 6000000.0, 20.0, 20.0)
    t2 = dummy_geotiff_factory(
        filename='tile_east.tif',
        width=20,
        height=20,
        bands=2,
        crs='EPSG:3161',
        transform=tf2,
        dtype=numpy.float32,
        nodata=-9999.0,
    )

    out_vrt = tmp_path / 'aligned_mosaic.vrt'
    mosaic.build_vrt_mosaic(
        input_paths=[t1, t2],
        output_path=out_vrt,
        target_crs='EPSG:3161',
        target_resolution=20.0,
    )

    with rasterio.open(out_vrt) as ds:
        assert ds.width == 40
        assert ds.height == 20
        assert ds.count == 2
        assert ds.nodata == -9999.0


def test_build_vrt_mosaic_categorical_and_relative_paths(
    dummy_geotiff_factory,
    tmp_path,
):
    '''
    Given: Categorical label rasters requesting relative source paths.
    When: Building the virtual mosaic with is_categorical=True.
    Then: Uses nearest resampling, sets nodata=255, and writes
        relative paths.
    '''
    t1 = dummy_geotiff_factory(
        filename='cat1.tif',
        width=10,
        height=10,
        bands=1,
        crs='EPSG:3161',
        dtype=numpy.uint8,
        nodata=255,
    )
    t2 = dummy_geotiff_factory(
        filename='cat2.tif',
        width=10,
        height=10,
        bands=1,
        crs='EPSG:3161',
        dtype=numpy.uint8,
        nodata=255,
    )

    out_vrt = tmp_path / 'categorical_mosaic.vrt'
    mosaic.build_vrt_mosaic(
        input_paths=[t1, t2],
        output_path=out_vrt,
        is_categorical=True,
        relative_paths=True,
    )

    tree = xml.etree.ElementTree.parse(out_vrt)
    source_filename_nodes = tree.findall('.//SourceFilename')
    assert len(source_filename_nodes) == 2
    for node in source_filename_nodes:
        assert node.attrib.get('relativeToVRT') == '1'
        assert node.text is not None
        assert not pathlib.Path(node.text).is_absolute()
