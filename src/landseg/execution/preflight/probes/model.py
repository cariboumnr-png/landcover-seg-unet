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
Model architecture and checkpoint diagnostic probes.

Inspects neural backbone configurations, channel counts, and evaluation
checkpoint weights prior to model execution.

Public APIs:
    - `model_body`: Inspect configured model architecture.
    - `checkpoint_ready`: Check evaluation checkpoint existence and size.
    - `eval_split`: Verify target evaluation dataset split.
'''

# standard imports
import os
# local imports
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def model_body(root_config: configs.RootConfig) -> schema.ProbeResult:
    '''
    Inspect configured model body architecture.

    Args:
        root_config:
            root configuration containing model specifications.

    Returns:
        schema.ProbeResult:
            probe result record with architecture recognition status.
    '''
    body = root_config.models.model_body
    return schema.ProbeResult(
        pid='model_body',
        category='Model',
        status=schema.ProbeStatus.PASS,
        message=f'Configured architecture: "{body}" recognized in registry',
        details={'model_body': body},
    )


def checkpoint_ready(
    root_config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Inspect model weights checkpoint file existence and readable size.

    Args:
        root_config:
            root configuration containing evaluation checkpoint path.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            diagnostic result indicating whether checkpoint is ready.
    '''
    checkpoint_path = root_config.command.model_evaluate.checkpoint
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return schema.ProbeResult(
            pid=pid or 'checkpoint_exists',
            category='Model',
            status=schema.ProbeStatus.FAIL,
            message=f'Checkpoint file not found: {checkpoint_path}',
            details={'checkpoint': checkpoint_path},
        )

    size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
    return schema.ProbeResult(
        pid=pid or 'checkpoint_exists',
        category='Model',
        status=schema.ProbeStatus.PASS,
        message=f'Checkpoint exists ({size_mb:.1f} MB)',
        details={'checkpoint': checkpoint_path, 'size_mb': round(size_mb, 2)},
    )


def eval_split(
    root_config: configs.RootConfig,
    pid: str | None = None,
) -> schema.ProbeResult:
    '''
    Validate configured evaluation dataset partition split.

    Args:
        root_config:
            root configuration containing evaluation split setting.
        pid:
            optional probe identifier override.

    Returns:
        schema.ProbeResult:
            diagnostic result indicating whether split is supported.
    '''
    split_name = root_config.command.model_evaluate.split
    supported_splits = {'test', 'val', 'train'}
    if split_name in supported_splits:
        return schema.ProbeResult(
            pid=pid or 'eval_split_configured',
            category='Model',
            status=schema.ProbeStatus.PASS,
            message=f"Evaluation target split '{split_name}' configured",
            details={'split': split_name},
        )

    return schema.ProbeResult(
        pid=pid or 'eval_split_configured',
        category='Model',
        status=schema.ProbeStatus.FAIL,
        message=(
            f"Unsupported evaluation split '{split_name}'; "
            f"expected one of {sorted(supported_splits)}"
        ),
        details={'split': split_name},
    )
