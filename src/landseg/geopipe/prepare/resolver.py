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
Resolution utilities for feature channels and target reclassification.

Parses user-specified preparation configurations (named schemes or
inline overrides) against ingested catalog schemas to resolve active
input feature channels and multi-head target hierarchies.
'''

# standard imports
import typing
# local imports
import landseg.geopipe.core as geo_core


# ----- `resolve_feature_channels`
def resolve_feature_channels(
    available_band_map: typing.Mapping[str, int],
    user_features_cfg: typing.Mapping[str, list[str] | str] | None,
    raster_schemes: typing.Mapping[str, typing.Mapping[str, list[str]]] | None,
) -> tuple[list[str], list[int]]:
    '''
    Resolve active feature band names and 0-based channel indices.

    If `user_features_cfg` is empty or None, all available bands in
    `available_band_map` are selected in sequential order.

    Args:
        available_band_map: Mapping of lower-case band names to 0-based
            channel indices in the ingested data blocks.
        user_features_cfg: Mapping of raster dataset or selection name to
            a scheme name (e.g. 'rgb_nir', 'all') or an explicit list of
            band names.
        raster_schemes: Mapping of raster dataset name to its named
            schemes dictionary from dataset manifest metadata.

    Returns:
        A tuple of (selected_band_names, selected_channel_indices).
    '''
    if not user_features_cfg:
        names = sorted(
            available_band_map.keys(), key=lambda k: available_band_map[k]
        )
        return names, [available_band_map[b] for b in names]

    selected_names: list[str] = []
    schemes_dict = raster_schemes or {}

    for raster_name, selection in user_features_cfg.items():
        if selection == 'all':
            matched = [
                b for b in available_band_map
                if b == raster_name or b.startswith(f'{raster_name}_')
            ]
            if not matched:
                matched = [b for b in available_band_map if b == raster_name]
            if not matched:
                raise ValueError(
                    f'No bands matching raster "{raster_name}" found in '
                    f'available bands: {list(available_band_map.keys())}'
                )
            selected_names.extend(matched)

        elif isinstance(selection, str):
            r_schemes = schemes_dict.get(raster_name, {})
            if not r_schemes and selection in schemes_dict:
                r_schemes = schemes_dict
            if selection not in r_schemes:
                raise ValueError(
                    f'Named feature scheme "{selection}" not found for '
                    f'raster "{raster_name}". Available: '
                    f'{list(r_schemes.keys())}'
                )
            scheme_bands = r_schemes[selection]
            for band in scheme_bands:
                if band not in available_band_map:
                    raise ValueError(
                        f'Band "{band}" from scheme "{selection}" not found '
                        f'in available bands: {list(available_band_map.keys())}'
                    )
            selected_names.extend(scheme_bands)

        elif isinstance(selection, list):
            for band in selection:
                if not isinstance(band, str):
                    raise TypeError(
                        f'Band name must be a string, got {type(band)}'
                    )
                if band not in available_band_map:
                    raise ValueError(
                        f'Band "{band}" in feature selection not found in '
                        f'available bands: {list(available_band_map.keys())}'
                    )
                selected_names.append(band)

        else:
            raise TypeError(
                f'Invalid feature selection type for "{raster_name}": '
                f'expected str or list of str, got {type(selection)}'
            )

    # deduplicate while preserving selection order
    deduped_names = [
        b for b in dict.fromkeys(selected_names) if b in available_band_map
    ]

    selected_indices = [available_band_map[b] for b in deduped_names]
    return deduped_names, selected_indices


# ----- `resolve_target_reclass`
def resolve_target_reclass(
    label_names_map: typing.Mapping[str, list[str]] | typing.Sequence[str],
    user_targets_cfg: typing.Mapping[str, str | geo_core.LabelScheme] | None,
    raster_schemes: typing.Mapping[str, typing.Mapping[str, geo_core.LabelScheme]] | None,
) -> dict[str, geo_core.LabelScheme | None]:
    '''
    Resolve active reclassification settings per target label layer.

    Args:
        label_names_map: Mapping of label layer name to list of class
            names or sequence of label layer names.
        user_targets_cfg: Mapping of label name to scheme name or explicit
            reclassification specification dictionary.
        raster_schemes: Mapping of raster name to named label schemes.

    Returns:
        Mapping of label layer name to resolved LabelScheme or None.
    '''
    resolved: dict[str, geo_core.LabelScheme | None] = {}
    if not user_targets_cfg:
        return {k: None for k in label_names_map}

    schemes_dict = raster_schemes or {}

    for label_name in label_names_map:
        cfg = user_targets_cfg.get(label_name)
        if cfg is None or cfg in ('raw', 'base', 'none'):
            resolved[label_name] = None
            continue

        if isinstance(cfg, str):
            r_schemes = schemes_dict.get(label_name, {})
            if not r_schemes and cfg in schemes_dict:
                r_schemes = schemes_dict
            if cfg not in r_schemes:
                raise ValueError(
                    f'Named target scheme "{cfg}" not found for label '
                    f'"{label_name}". Available: {list(r_schemes.keys())}'
                )
            resolved[label_name] = dict(r_schemes[cfg])  # type: ignore[assignment]

        elif isinstance(cfg, dict):
            if 'reclass' not in cfg or not isinstance(cfg['reclass'], dict):
                raise ValueError(
                    f'Target reclassification dict for "{label_name}" must '
                    'contain a "reclass" mapping'
                )
            resolved[label_name] = cfg

        else:
            raise TypeError(
                f'Invalid target config type for "{label_name}": '
                f'expected str or dict, got {type(cfg)}'
            )

    return resolved
