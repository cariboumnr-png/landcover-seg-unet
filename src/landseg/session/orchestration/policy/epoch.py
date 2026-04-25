# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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
Epoch level orchestration policy.
'''

# standard imports
import typing
# local imports
import landseg.session.engine as engine
import landseg.session.orchestration.event as events

class EpochPolicy:
    '''doc'''

    def __init__(
        self,
        *,
        training_engine: engine.TrainingEpochRunner,
        phase_name: str,
        epoch_index: int,
        active_heads: list[str] | None = None
    ):
        '''doc'''

        self.epoch = epoch_index
        self.phase = phase_name
        self.engine = training_engine
        self.active_heads = active_heads

    def run(self) -> typing.Generator[events.Event, None, dict[str, float]]:
        '''doc'''
        # epoch starts
        yield events.EpochStart(self.epoch, self.phase)

        # delegate execution to epoch runner
        epoch_metrics = self.engine.run(self.epoch)

        # parse epoch metrics into dict
        parsed_metrics = self._parse_metrics(epoch_metrics)

        # epoch ends
        yield events.EpochEnd(self.epoch, self.phase, parsed_metrics)

        # enables downstream `yield from`
        return parsed_metrics

    def execute(self):
        '''Run the underlying engine and return raw metrics.'''
        return self.engine.run(self.epoch)

    def _parse_metrics(self, metrics: engine.EpochMetrics) -> dict[str, float]:
        '''Helper to parse epoch metrics into a plain dictionary.'''

        train_losses = metrics.training
        val_accuracies = metrics.validation or {}
        parsed: dict[str, float] = {}
        # get heads metrics
        val_heads = list(val_accuracies.keys())
        # get active heads for metrics extraction
        active = val_heads if self.active_heads is None else self.active_heads
        # add loss values to snapshot
        parsed.update(**train_losses)
        # accuracies from active heads
        miou_sum = ac_miou_sum = 0.0
        for h in active:
            ious = val_accuracies.get(h, {})
            parsed[f'miou_{h}'] = ious.get('mean', 0.0)
            miou_sum += parsed[f'miou_{h}']
            parsed[f'ac_miou_{h}'] = ious.get('ac_mean', 0.0)
            ac_miou_sum += parsed[f'ac_miou_{h}']
        # mean accuraries from heads
        parsed['mean_iou_active_heads'] = miou_sum / len(active)
        parsed['mean_ac_iou_active_heads'] = ac_miou_sum / len(active)
        return parsed
