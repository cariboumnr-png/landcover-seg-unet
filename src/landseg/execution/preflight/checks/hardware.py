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
Hardware and compute environment diagnostic checks.

Inspects accelerator device availability, PyTorch runtime version, and
estimated batch memory headroom.

Public APIs:
    - `hardware_info`: Inspect accelerator and compute environment.
'''

# third-party imports
import torch
# local imports
import landseg._constants as c
import landseg.execution.preflight.context as context
import landseg.execution.preflight.schema as schema


# ----- public functions
def hardware_info(ctx: context.InspectionContext) -> list[schema.ProbeResult]:
    '''Inspect accelerator availability and estimate batch VRAM headroom.'''
    results: list[schema.ProbeResult] = []

    if not torch.cuda.is_available():
        results.append(
            schema.ProbeResult(
                pid='cuda_device',
                category=schema.ProbeCategory.HARDWARE,
                status=schema.ProbeStatus.WARN,
                message='CUDA unavailable; compute running on CPU',
                details={
                    'device_name': 'cpu',
                    'cuda': False,
                    'device': c.DEVICE_NAME,
                    'torch_version': torch.__version__,
                },
            )
        )
        return results

    device_name = torch.cuda.get_device_name(0)
    results.append(
        schema.ProbeResult(
            pid='cuda_device',
            category=schema.ProbeCategory.HARDWARE,
            status=schema.ProbeStatus.PASS,
            message=f'{device_name} (cuda:0)',
            details={
                'device_name': device_name,
                'cuda': True,
                'device': c.DEVICE_NAME,
                'torch_version': torch.__version__,
            },
        )
    )

    try:
        free_b, total_b = torch.cuda.mem_get_info(0)
    except Exception:  # pylint: disable=broad-exception-caught
        return results

    batch_size = _resolve_batch_size(ctx)
    # coarse estimate: B * C * H * W * 4 bytes * 10x overhead factor
    est_b = batch_size * 4 * 256 * 256 * 4 * 10
    free_gb = free_b / (1024 ** 3)
    est_gb = est_b / (1024 ** 3)
    results.append(
        schema.ProbeResult(
            pid='vram_headroom',
            category=schema.ProbeCategory.HARDWARE,
            status=(
                schema.ProbeStatus.PASS
                if free_b >= est_b
                else schema.ProbeStatus.WARN
            ),
            message=f'{free_gb:.1f} GB free / ~{est_gb:.1f} GB est. batch',
            details={
                'vram_free_gb': round(free_gb, 2),
                'vram_total_gb': round(total_b / (1024 ** 3), 2),
                'estimated_batch_gb': round(est_gb, 2),
            },
        )
    )
    return results


# ----- private helpers
def _resolve_batch_size(ctx: context.InspectionContext) -> int:
    '''Resolve configured batch size, falling back to a sane default.'''
    config = ctx.config
    if hasattr(config, 'session') and hasattr(config.session, 'dataloader'):
        return int(config.session.dataloader.batch_size)
    return 16
