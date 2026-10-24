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

'''
Multi-raster channel composition and stacking operations.

This module provides functionality to combine multiple feature and label
rasters into unified multi-band GDAL Virtual Raster (VRT) composites.

Public APIs:
    - `stack_rasters`: Stack feature and label rasters into composite VRTs.
'''

# standard imports
import pathlib
import typing
# local imports
import landseg.geopipe.rasters as rasters


# ----- public functions
def compose_rasters(
    features_fpaths: list[pathlib.Path],
    labels_fpaths: list[pathlib.Path],
    output_dir: pathlib.Path,
) -> typing.Generator[str, None, dict[str, pathlib.Path]]:
    '''
    Stack feature and label rasters into composite VRT files.

    Args:
        features_fpaths:
            List of file paths for feature rasters.
        labels_fpaths:
            List of file paths for label rasters.
        output_dir:
            Directory path where stacked VRT files will be saved.

    Yields:
        str:
            Status messages during stacking process.

    Returns:
        dict[str, pathlib.Path]:
            Mapping of raster role ('features', 'labels') to stacked
            VRT file path.
    '''
    output_dir = pathlib.Path(output_dir)
    stacked: dict[str, pathlib.Path] = {}
    yield 'Stacking rasters if applicable'

    # features
    n = len(features_fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'features': features_fpaths[0]})
    else:
        out_path = output_dir / 'harmonized_features_STACKED.vrt'
        dataset_meta = _read_feature_metadata(features_fpaths)
        rasters.stack_rasters(features_fpaths, out_path, dataset_tags=dataset_meta)
        stacked.update({'features': out_path})
        yield f'Feature rasters stacked to {out_path} (n={n})'

    # labels
    n = len(labels_fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'labels': labels_fpaths[0]})
    else:
        out_path = output_dir / 'harmonized_labels_STACKED.vrt'
        bands_meta = _read_label_band_metadata(labels_fpaths)
        rasters.stack_rasters(labels_fpaths, out_path, band_tags=bands_meta)
        stacked.update({'labels': out_path})
        yield f'Label rasters stacked to {out_path} (n={n})'

    return stacked


# ----- private helpers
def _read_feature_metadata(paths: list[pathlib.Path]) -> dict[str, typing.Any]:
    '''Collect and merge feature-scheme metadata (dataset tags).'''
    merged_schema: dict[str, typing.Any] = {}
    for path in paths:
        tags = rasters.get_info(path).tags
        schemes_as_str = tags.get('schemes')

        if schemes_as_str is None:
            continue

        schemes_dict = rasters.parse_vrt_tag(schemes_as_str)
        if not isinstance(schemes_dict, dict):
            continue

        for key, val in schemes_dict.items():
            if isinstance(val, dict):
                target = merged_schema.setdefault(key, {})
                for scheme_name, bands in val.items():
                    existing = target.get(scheme_name)
                    if existing is not None and existing != bands:
                        raise ValueError(
                            f'Conflicting definitions for feature scheme '
                            f'"{scheme_name}" under raster "{key}": '
                            f'{existing!r} != {bands!r}'
                        )
                    target[scheme_name] = bands
            elif isinstance(val, list):
                existing = merged_schema.get(key)
                if existing is not None and existing != val:
                    raise ValueError(
                        f'Conflicting definitions for feature scheme "{key}": '
                        f'{existing!r} != {val!r}'
                    )
                merged_schema[key] = val

    return {'schemes': merged_schema}


def _read_label_band_metadata(paths: list[pathlib.Path]) -> list[dict[str, str]]:
    '''Convert label dataset metadata into composite band metadata.'''
    metadata: list[dict[str, str]] = []
    for path in paths:
        with rasters.read_raster(path) as source:
            dataset_tags = source.tags()
            for band_index in source.indexes:
                # band's own tags take precedence over dataset-level tags.
                tags = {**dataset_tags, **source.tags(band_index)}
                metadata.append(tags)
    return metadata
