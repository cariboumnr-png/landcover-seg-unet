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
Hardware and compute environment diagnostic probes.

Inspects accelerator device availability, PyTorch runtime version, and
hardware capabilities.

Public APIs:
    - `hardware_info`: Check accelerator and compute environment.
'''

# third-party imports
import torch
# local imports
import landseg._constants as c
import landseg.configs as configs
import landseg.execution.preflight.schema as schema


# ----- public functions
def hardware_info(
    root_config: configs.RootConfig,
) -> list[schema.ProbeResult]:
    '''
    Inspect compute accelerator availability and system hardware state.

    Args:
        root_config:
            root configuration for GPU check and batch memory estimate.

    Returns:
        list[schema.ProbeResult]:
            list of hardware diagnostic probe results.
    '''
    probes: list[schema.ProbeResult] = []
    cuda_available = torch.cuda.is_available()

    if root_config.command.preflight.check_gpu:
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            probes.append(
                schema.ProbeResult(
                    pid='cuda_device',
                    category='Hardware',
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
                free_gb = free_b / (1024 ** 3)
                total_gb = total_b / (1024 ** 3)
                batch_size = (
                    root_config.session.dataloader.batch_size
                    if root_config is not None
                    and hasattr(root_config, 'session')
                    and hasattr(root_config.session, 'dataloader')
                    else 16
                )
                # estimate B * C * H * W * 4 bytes * 10x overhead factor
                est_b = batch_size * 4 * 256 * 256 * 4 * 10
                est_gb = est_b / (1024 ** 3)
                vram_status = (
                    schema.ProbeStatus.PASS
                    if free_b >= est_b
                    else schema.ProbeStatus.WARN
                )
                probes.append(
                    schema.ProbeResult(
                        pid='vram_headroom',
                        category='Hardware',
                        status=vram_status,
                        message=(
                            f'{free_gb:.1f} GB free / '
                            f'~{est_gb:.1f} GB est. batch'
                        ),
                        details={
                            'vram_free_gb': round(free_gb, 2),
                            'vram_total_gb': round(total_gb, 2),
                            'estimated_batch_gb': round(est_gb, 2),
                            'vram_free_bytes': free_b,
                            'total_bytes': total_b,
                            'estimated_batch_bytes': est_b,
                        },
                    )
                )
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        else:
            probes.append(
                schema.ProbeResult(
                    pid='cuda_device',
                    category='Hardware',
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

    return probes
