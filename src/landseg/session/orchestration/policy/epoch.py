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

    def run(self) -> typing.Generator[events.Event, None, engine.EpochMetrics]:
        '''doc'''
        # epoch starts
        yield events.EpochStart(self.epoch, self.phase)

        # delegate execution to epoch runner
        epoch_metrics = self.engine.run(self.epoch)

        # epoch ends
        yield events.EpochEnd(self.epoch, self.phase, epoch_metrics)

        # enables downstream `yield from`
        return epoch_metrics

    def execute(self):
        '''Run the underlying engine and return raw metrics.'''
        return self.engine.run(self.epoch)
