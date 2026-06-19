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

'''Architecture preset objectives.'''

# third-party imports
import optuna
# local imports
import landseg.study.sweep as sweep

def obj_architecture(
    trial_cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Architecture preset mutations:
      - Model body (`str`)
      - Base channel count (`int`)
      - Bottleneck type (`str`)
    '''

    study_cfg = trial_cfg.study.architecture

    trial_cfg.set_model_body(
        model_body=trial.suggest_categorical(
            name='model.model_body',
            choices=study_cfg.model_body,
        )
    )

    trial_cfg.set_model_base_channel(
        base_channel=trial.suggest_int(
            name='model.base_channel',
            low=study_cfg.base_channel[0],
            high=study_cfg.base_channel[1],
            step=study_cfg.base_channel[2],
        )
    )

    trial_cfg.set_model_bottleneck(
        bottleneck=trial.suggest_categorical(
            name='model.bottleneck',
            choices=study_cfg.bottleneck,
        )
    )

    return trial_cfg

def obj_bottleneck(
    trial_cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Bottlenect preset mutations:
      - Blocks count (`int`)
      - Transformer hyperparameters
    '''

    study_cfg = trial_cfg.study.bottleneck

    bottleneck = trial.suggest_categorical(
        name='model.bottleneck',
        choices=study_cfg.bottleneck,
    )
    trial_cfg.set_model_bottleneck(bottleneck)

    # depending on bottleneck type, sample relevant parameters
    if bottleneck in ('conv', 'hybrid'):
        trial_cfg.set_bottleneck_convolution_blocks(
            num_blocks=trial.suggest_int(
                name='bottleneck.num_conv_blocks',
                low=study_cfg.num_conv_blocks[0],
                high=study_cfg.num_conv_blocks[1],
                step=study_cfg.num_conv_blocks[2],
            )
        )
    if bottleneck in ('transformer', 'hybrid'):
        trial_cfg.set_bottleneck_transformer_blocks(
            num_blocks=trial.suggest_int(
                name='bottleneck.num_transformer_blocks',
                low=study_cfg.num_transformer_blocks[0],
                high=study_cfg.num_transformer_blocks[1],
                step=study_cfg.num_transformer_blocks[2],
            )
        )
        trial_cfg.set_transformer_num_heads(
            num_heads=trial.suggest_categorical(
                name='transformer.num_heads',
                choices=study_cfg.num_heads,
            )
        )
        trial_cfg.set_transformer_mlp_ratio(
            mlp_ratio=trial.suggest_float(
                name='transformer.mlp_ratio',
                low=study_cfg.mlp_ratio[0],
                high=study_cfg.mlp_ratio[1],
            )
        )
        trial_cfg.set_transformer_dropout(
            dropout=trial.suggest_float(
                name='transformer.dropout',
                low=study_cfg.dropout[0],
                high=study_cfg.dropout[1],
            )
        )
        trial_cfg.set_transformer_attn_dropout(
            attn_dropout=trial.suggest_float(
                name='transformer.attn_dropout',
                low=study_cfg.attn_dropout[0],
                high=study_cfg.attn_dropout[1],
            )
        )

    return trial_cfg

def obj_conditioning(
    trial_cfg: sweep.RootConfigShape,
    trial: optuna.Trial,
) -> sweep.RootConfigShape:
    '''
    Conditioning preset mutations:
      - Conditioner selection (`list[str]`)
    '''

    study_cfg = trial_cfg.study.conditioning

    # auto construct conditioners mapping
    # e.g., {'film_concat': ['film', 'concat']}
    choice_map = {}
    for conditioners in study_cfg.conditioners:
        if not conditioners:
            key = "none"
        else:
            key = "_".join(conditioners)
        choice_map[key] = conditioners

    selected = trial.suggest_categorical(
        "model.conditioners",
        list(choice_map.keys()),
    )

    # reconstruct list of strs from mapping
    # e.g., 'film_concat' -> ['film', 'concat']
    trial_cfg.set_model_conditioners(
        choice_map[selected]
    )

    return trial_cfg
