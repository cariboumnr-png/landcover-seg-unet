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
    - `probe_hardware`: Check accelerator and compute environment.
'''

# standard imports
import typing
# third-party imports
import torch
# local imports
import landseg._constants as c
import landseg.execution.preflight.schema as schema


# ----- public functions
def probe_hardware(
    check_gpu: bool = True,
    telemetry: dict[str, typing.Any] | None = None,
    root_config: typing.Any = None,
) -> list[schema.ProbeResult]:
    '''
    Inspect compute accelerator availability and system hardware state.

    Args:
        check_gpu:
            whether GPU accelerator availability should be verified.
        telemetry:
            optional dictionary to populate with hardware metadata.
        root_config:
            optional root configuration for batch tensor memory estimate.

    Returns:
        list[schema.ProbeResult]:
            list of hardware diagnostic probe results.
    '''
    probes: list[schema.ProbeResult] = []
    cuda_available = torch.cuda.is_available()

    if telemetry is not None:
        telemetry['torch_version'] = torch.__version__
        telemetry['device'] = c.DEVICE_NAME
        telemetry['cuda_available'] = cuda_available

    if check_gpu:
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            probes.append(
                schema.ProbeResult(
                    probe_id='cuda_device',
                    category='hardware',
                    status=schema.ProbeStatus.PASS,
                    message=f'{device_name} (cuda:0)',
                    details={'device_name': device_name, 'cuda': True},
                )
            )
            try:
                free_b, total_b = torch.cuda.mem_get_info(0)
                free_gb = free_b / (1024 ** 3)
                batch_size = (
                    root_config.session.data_loader.batch_size
                    if root_config is not None
                    and hasattr(root_config, 'session')
                    and hasattr(root_config.session, 'data_loader')
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
                        probe_id='vram_headroom',
                        category='hardware',
                        status=vram_status,
                        message=(
                            f'{free_gb:.1f} GB free / '
                            f'~{est_gb:.1f} GB est. batch'
                        ),
                        details={
                            'vram_free_bytes': free_b,
                            'total_bytes': total_b,
                            'estimated_batch_bytes': est_b,
                        },
                    )
                )
                if telemetry is not None:
                    telemetry['vram_free_gb'] = round(free_gb, 2)
                    total_gb = total_b / (1024 ** 3)
                    telemetry['vram_total_gb'] = round(total_gb, 2)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
        else:
            probes.append(
                schema.ProbeResult(
                    probe_id='cuda_device',
                    category='hardware',
                    status=schema.ProbeStatus.WARN,
                    message='CUDA unavailable; compute running on CPU',
                    details={'device_name': 'cpu', 'cuda': False},
                )
            )

    return probes
