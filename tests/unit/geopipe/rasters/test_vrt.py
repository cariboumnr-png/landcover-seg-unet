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

'''Unit tests for shared GDAL Virtual Raster (VRT) utility primitives.'''

# standard imports
import xml.etree.ElementTree
# third-party imports
import numpy
import pytest
import rasterio
import rasterio.crs
import rasterio.transform
# local imports
import landseg.geopipe.rasters.mask as mask
import landseg.geopipe.rasters.vrt as vrt


# ----- `gdal_dtype_name` tests
def test_gdal_dtype_name():
    '''
    Given: Rasterio and numpy data type representations.
    When: Translating through `gdal_dtype_name`.
    Then: Produces canonical GDAL data type names.
    '''
    assert vrt.gdal_dtype_name('uint8') == 'Byte'
    assert vrt.gdal_dtype_name(numpy.uint8) == 'Byte'
    assert vrt.gdal_dtype_name('float32') == 'Float32'
    assert vrt.gdal_dtype_name('int16') == 'Int16'
    assert vrt.gdal_dtype_name('custom_type') == 'Custom_type'


# ----- VRT XML construction tests
def test_create_vrt_root_and_write(tmp_path):
    '''
    Given: Raster spatial dimensions, CRS, and affine transform.
    When: Creating root with `create_vrt_root` and serializing.
    Then: Writes a valid VRT XML dataset file with exact attributes.
    '''
    crs = rasterio.crs.CRS.from_string('EPSG:3161')
    transform = rasterio.transform.from_origin(
        500000.0, 6000000.0, 20.0, 20.0
    )
    root = vrt.create_vrt_root(
        width=100,
        height=50,
        crs=crs,
        transform=transform,
    )

    out_file = tmp_path / 'test_root.vrt'
    result_path = vrt.write_vrt(root, out_file)
    assert result_path == out_file
    assert out_file.exists()

    tree = xml.etree.ElementTree.parse(out_file)
    dataset = tree.getroot()
    assert dataset.attrib.get('rasterXSize') == '100'
    assert dataset.attrib.get('rasterYSize') == '50'
    assert dataset.find('SRS') is not None
    assert dataset.find('GeoTransform') is not None


def test_add_vrt_band_and_source(tmp_path):
    '''
    Given: An initialized VRT dataset root element.
    When: Appending a band with `add_vrt_band` and a source.
    Then: Inserts correctly configured band and source XML nodes.
    '''
    crs = rasterio.crs.CRS.from_string('EPSG:3161')
    transform = rasterio.transform.from_origin(
        500000.0, 6000000.0, 20.0, 20.0
    )
    root = vrt.create_vrt_root(10, 10, crs, transform)

    band_node = vrt.add_vrt_band(
        root=root,
        band_idx=1,
        dtype='float32',
        nodata=-9999.0,
        description='spectral_b1',
    )
    assert band_node.attrib.get('dataType') == 'Float32'
    nodata_node = band_node.find('NoDataValue')
    assert nodata_node is not None
    assert nodata_node.text == '-9999.0'
    desc_node = band_node.find('Description')
    assert desc_node is not None
    assert desc_node.text == 'spectral_b1'

    src_node = vrt.add_vrt_source(
        band_node=band_node,
        source_path=tmp_path / 'dummy.tif',
        source_band=1,
        width=10,
        height=10,
        dtype='float32',
        nodata=-9999.0,
        relative_paths=True,
        output_dir=tmp_path,
    )
    assert src_node.tag == 'ComplexSource'
    filename_node = src_node.find('SourceFilename')
    assert filename_node is not None
    assert filename_node.attrib.get('relativeToVRT') == '1'
    band_elem = src_node.find('SourceBand')
    assert band_elem is not None
    assert band_elem.text == '1'
    nodata_elem = src_node.find('NODATA')
    assert nodata_elem is not None
    assert nodata_elem.text == '-9999.0'


# ----- VRT metadata and description tests
def test_vrt_metadata_and_descriptions(dummy_geotiff_factory, tmp_path):
    '''
    Given: An initialized Virtual Raster (.vrt) dataset.
    When: Setting band descriptions and dataset-level metadata tags.
    Then: Persist descriptions and tags cleanly into the VRT file.
    '''
    comp_path = dummy_geotiff_factory(
        filename='comp_meta.tif',
        width=20,
        height=20,
        bands=2,
        dtype=numpy.uint8,
    )
    vrt_path = tmp_path / 'test_meta.vrt'
    mask.unify_nodata_mask(comp_path, vrt_path)

    vrt.add_band_description_to_vrt(vrt_path, {1: 'mask_channel'})
    vrt.add_tag_to_vrt(vrt_path, sensor='sentinel2', processed='true')
    # verify empty tags call returns without modification
    vrt.add_tag_to_vrt(vrt_path, empty_val=None)

    with rasterio.open(vrt_path) as dst:
        assert dst.descriptions[0] == 'mask_channel'
        tags = dst.tags()
        assert tags['sensor'] == 'sentinel2'
        assert tags['processed'] == 'true'

    with pytest.raises(ValueError, match='Expected 1 band descriptions'):
        vrt.add_band_description_to_vrt(vrt_path, {1: 'b1', 2: 'b2'})
