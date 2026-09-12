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
Block normalization utilities.

Applies global image normalization to raw data blocks using statistics
computed from training data. Produces normalized block artifacts and
maintains split-indexed file mappings for downstream schema generation.
'''

# standard imports
import typing
# third-party imports
import numpy
# local imports
import landseg.geopipe.core as geo_core


def reclassify_label_stack(
    raw_labels: numpy.ndarray | typing.Sequence[numpy.ndarray],
    label_layer_names: typing.Sequence[str],
    target_reclass: typing.Mapping[str, geo_core.LabelScheme | None],
    ignore_index: int,
) -> numpy.ndarray:
    '''
    Build a multi-head label stack applying active target reclassifications.

    Args:
        raw_labels: 3D array of shape [L, H, W] or list of 2D arrays.
        label_layer_names: Names corresponding to each base label layer.
        target_reclass: Mapping of label layer name to reclass config.
        ignore_index: Integer index for invalid/masked pixels (e.g. 255).

    Returns:
        A 3D numpy array of shape [L, H, W] containing the transformed stack.
    '''
    if isinstance(raw_labels, numpy.ndarray):
        if raw_labels.ndim == 3:
            label_list = [raw_labels[i] for i in range(raw_labels.shape[0])]
        elif raw_labels.ndim == 2:
            label_list = [raw_labels]
        else:
            raise ValueError(
                f'Expected 2D or 3D label array, got shape {raw_labels.shape}'
            )
    else:
        label_list = list(raw_labels)

    stack: list[numpy.ndarray] = []

    for i, arr in enumerate(label_list):
        name = (
            label_layer_names[i]
            if i < len(label_layer_names)
            else f'label_{i}'
        )
        reclass_cfg = target_reclass.get(name)

        if not reclass_cfg or not reclass_cfg.get('reclass'):
            stack.append(arr)
            continue

        reclass = reclass_cfg['reclass']
        # 1. Base layer
        stack.append(arr)

        # 2. Child slices
        group_layer = numpy.full_like(arr, ignore_index, dtype=arr.dtype)
        for group_id, classes in reclass.items():
            mask = numpy.isin(arr, classes)
            group_layer[mask] = int(group_id)

            child_arr = numpy.where(mask, arr, ignore_index)
            for k, cls_id in enumerate(classes, 1):
                child_arr[child_arr == cls_id] = int(k)
            stack.append(child_arr)

        # 3. Grouping layer
        stack.append(group_layer)

    return numpy.stack(stack, axis=0)
