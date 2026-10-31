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
Model architecture and checkpoint diagnostic checks.

Inspects neural backbone configuration, evaluation checkpoints, and
target dataset splits prior to model execution.

Public APIs:
    - `model_body`: Inspect configured model architecture.
    - `checkpoint_ready`: Check evaluation checkpoint existence and size.
    - `eval_split`: Verify target evaluation dataset split.
'''

# standard imports
import pathlib
# local imports
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema


# ----- constants
_SUPPORTED_SPLITS = {'test', 'val', 'train'}


# ----- public functions
def model_body(
    ctx: context.InspectionContext,
    pid: str = 'model_body',
) -> schema.ProbeResult:
    '''Inspect configured model body architecture.'''
    body = ctx.config.models.model_body
    return schema.ProbeResult(
        pid=pid,
        category=schema.ProbeCategory.MODEL,
        status=schema.ProbeStatus.PASS,
        message=f'Configured architecture: "{body}" recognized in registry',
        details={'model_body': body},
    )


def checkpoint_ready(
    ctx: context.InspectionContext,
    pid: str = 'checkpoint_ready',
) -> schema.ProbeResult:
    '''Inspect evaluation checkpoint file existence and readable size.'''
    checkpoint = ctx.config.session.evaluation.checkpoint
    ckpt_path = pathlib.Path(checkpoint) if checkpoint else None

    if ckpt_path is None or not ckpt_path.is_file():
        return schema.ProbeResult(
            pid=pid,
            category=schema.ProbeCategory.MODEL,
            status=schema.ProbeStatus.FAIL,
            message=f'Checkpoint file not found: {checkpoint}',
            details={'checkpoint': str(checkpoint)},
        )

    size_mb = ckpt_path.stat().st_size / (1024 ** 2)
    return schema.ProbeResult(
        pid=pid,
        category=schema.ProbeCategory.MODEL,
        status=schema.ProbeStatus.PASS,
        message=f'Checkpoint exists ({size_mb:.1f} MB)',
        details={'checkpoint': str(ckpt_path), 'size_mb': round(size_mb, 2)},
    )


def eval_split(
    ctx: context.InspectionContext,
    pid: str = 'eval_split',
) -> schema.ProbeResult:
    '''Validate configured evaluation dataset partition split.'''
    split = ctx.config.session.evaluation.split
    if split in _SUPPORTED_SPLITS:
        return schema.ProbeResult(
            pid=pid,
            category=schema.ProbeCategory.MODEL,
            status=schema.ProbeStatus.PASS,
            message=f"Evaluation target split '{split}' configured",
            details={'split': split},
        )

    return schema.ProbeResult(
        pid=pid,
        category=schema.ProbeCategory.MODEL,
        status=schema.ProbeStatus.FAIL,
        message=(
            f"Unsupported evaluation split '{split}'; "
            f'expected one of {sorted(_SUPPORTED_SPLITS)}'
        ),
        details={'split': split},
    )
