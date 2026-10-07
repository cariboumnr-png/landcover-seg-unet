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
Dataset partition and feature contract diagnostic probes.

Inspects train/validation/test split ratios and feature/target contracts.

Public APIs:
    - `split_ratios`: Validate dataset partition split ratio proportions.
    - `dataset_targets`: Inspect configured features and target mappings.
'''

# local imports
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def split_ratios(
    config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate dataset train, validation, and test split ratios.

    Args:
        config:
            root configuration containing dataset partition parameters.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            diagnostic probe record evaluating split ratio proportions.
    '''
    probe_id = pid or 'split_ratios'
    partition = config.data.preparation.partition
    val_r = partition.val_ratio
    test_r = partition.test_ratio
    train_r = 1.0 - (val_r + test_r)

    if val_r < 0.0 or test_r < 0.0 or train_r < 0.0:
        return schema.ProbeResult(
            pid=probe_id,
            category='Dataset',
            status=schema.ProbeStatus.FAIL,
            message=(
                f'Invalid split ratios: val={val_r:.2f}, test={test_r:.2f} '
                f'exceed total ratio of 1.0'
            ),
            details={'train': train_r, 'val': val_r, 'test': test_r},
        )

    return schema.ProbeResult(
        pid=probe_id,
        category='Dataset',
        status=schema.ProbeStatus.PASS,
        message=(
            f'train: {train_r:.2f} | val: {val_r:.2f} | '
            f'test: {test_r:.2f}'
        ),
        details={'train': train_r, 'val': val_r, 'test': test_r},
    )


def dataset_targets(
    config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Inspect configured dataset features and target mappings.

    Args:
        config:
            root configuration containing preparation specifications.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            diagnostic probe record for feature and target specifications.
    '''
    probe_id = pid or 'dataset_targets'
    prep = config.data.preparation
    feat_count = len(prep.features)
    target_count = len(prep.targets)

    if feat_count == 0 or target_count == 0:
        return schema.ProbeResult(
            pid=probe_id,
            category='Dataset',
            status=schema.ProbeStatus.WARN,
            message=(
                f'Features ({feat_count}) or targets ({target_count}) '
                'empty; relying on defaults'
            ),
            details={
                'features_count': feat_count,
                'targets_count': target_count,
            },
        )

    return schema.ProbeResult(
        pid=probe_id,
        category='Dataset',
        status=schema.ProbeStatus.PASS,
        message=(
            f'{feat_count} feature source(s) and {target_count} target(s)'
        ),
        details={
            'features': list(prep.features.keys()),
            'targets': list(prep.targets.keys()),
        },
    )
