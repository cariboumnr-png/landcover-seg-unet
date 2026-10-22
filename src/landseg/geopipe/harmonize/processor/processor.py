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

# pylint: disable=missing-function-docstring

'''
Data harmonization pipeline command implementation.
'''

# standard imports
import dataclasses
import pathlib
import typing
# third-party imports
import rasterio
import rasterio.crs
import rasterio.enums
import rasterio.transform
# local imports
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.processor.composition as composition
import landseg.geopipe.rasters as rasters


# ----- public dataclasses
@dataclasses.dataclass
class HarmonizedRasters:
    '''Container for processed raster paths dictionaries.'''
    provenance: dict[str, pathlib.Path] = dataclasses.field(default_factory=dict)
    harmonized: dict[str, pathlib.Path] = dataclasses.field(default_factory=dict)
    finalized: dict[str, pathlib.Path] = dataclasses.field(default_factory=dict)


# ----- private dataclasses
@dataclasses.dataclass
class _ParsedManifestEntry:
    file_path: pathlib.Path
    tagged_name: str
    raster_type: str
    raster_band_mapping: dict[int, str]
    resampling_method: rasterio.enums.Resampling


# ----- public functions
def harmonizer(  # pylint: disable=too-many-arguments, too-many-locals
    mfst_entries: list[manifest.ManifestEntry],
    output_dir: pathlib.Path,
    *,
    dst_crs: rasterio.crs.CRS,
    dst_transform: rasterio.transform.Affine,
    dst_width: int,
    dst_height: int,
    categorical_resampling: str,
    continuous_resampling: str,
) -> typing.Generator[str, None, HarmonizedRasters]:
    '''Harmonize all compiled raster sources onto the canonical grid.'''
    features: list[pathlib.Path] = []
    labels: list[pathlib.Path] = []
    processed = HarmonizedRasters()

    for mfst in mfst_entries:

        parsed = _parse_manifest(
            mfst,
            categorical_resampling,
            continuous_resampling
        )
        out_vrt = output_dir / f'{parsed.tagged_name}.vrt'

        yield (
            f'Harmonizing raster {parsed.file_path} -> {out_vrt} '
            f'(resampling: {parsed.resampling_method.name})'
        )

        # warp and write
        with rasterio.open(parsed.file_path) as src:
            warped = rasters.write_warped_vrt(
                src=src,
                output_path=out_vrt,
                crs=dst_crs,
                transform=dst_transform,
                height=dst_height,
                width=dst_width,
                resampling=parsed.resampling_method,
                nodata=src.nodata
            )

        match parsed.raster_type:
            case 'domains' | 'domain':
                _tag_domain_metadata(warped, mfst)
                processed.finalized[parsed.tagged_name] = warped
            case 'features' | 'feature':
                _tag_feature_metadata(warped, mfst)
                features.append(warped)
            case 'labels' | 'label':
                _tag_label_metadata(warped, mfst)
                labels.append(warped)

        processed.provenance[parsed.tagged_name] = parsed.file_path.resolve()
        processed.harmonized[parsed.tagged_name] = warped.resolve()

        rasters.add_band_description_to_vrt(warped, parsed.raster_band_mapping)

    processed.finalized.update(
        **(yield from composition.compose_rasters(features, labels, output_dir))
    )

    return processed


# ----- private helpers
def _parse_manifest(
    mfst: manifest.ManifestEntry,
    categorical_resampling: str,
    continuous_resampling: str,
) -> _ParsedManifestEntry:
    '''Parse manifest entry dict into a typed container.'''
    is_cat = mfst['category'] in {'domains', 'domain', 'labels', 'label'}
    resampling = categorical_resampling if is_cat else continuous_resampling
    return _ParsedManifestEntry(
        file_path=mfst['path'],
        tagged_name=f'{mfst['category']}_{mfst["name"]}',
        raster_type=mfst['category'],
        raster_band_mapping=mfst['band_mapping'],
        resampling_method=rasters.resolve_resampling_algorithm(resampling, is_cat)
    )


def _tag_domain_metadata(
    warped: pathlib.Path,
    mfst: manifest.ManifestEntry,
) -> None:
    '''Attach domain raster metadata tags to VRT file.'''
    cat_specs = mfst.get('categorical_specs')
    if not cat_specs:
        return

    if 'index_base' in cat_specs:
        rasters.add_tag_to_vrt(warped, index_base=cat_specs['index_base'])


def _tag_feature_metadata(
    warped: pathlib.Path,
    mfst: manifest.ManifestEntry,
) -> None:
    '''Attach feature schemes metadata tags to VRT file.'''
    schemes = mfst.get('schemes')
    if schemes:
        rasters.add_tag_to_vrt(warped, schemes={mfst['name']: schemes})


def _tag_label_metadata(
    warped: pathlib.Path,
    mfst: manifest.ManifestEntry,
) -> None:
    '''Attach categorical label metadata tags to VRT file.'''
    cat_specs = mfst.get('categorical_specs')
    if not cat_specs:
        raise ValueError('Missing categorical specs for label raster')

    schemes = mfst.get('schemes')
    rasters.add_tag_to_vrt(
        warped,
        index_base=cat_specs['index_base'],
        num_cls=cat_specs['num_cls'],
        ignore_cls=cat_specs['ignore_cls'],
        class_name=cat_specs.get('class_name', {}),
        color_map=cat_specs.get('color_map', {}),
        taxonomy=cat_specs.get('taxonomy', {}),
        schemes={mfst['name']: schemes} if schemes else {},
    )
