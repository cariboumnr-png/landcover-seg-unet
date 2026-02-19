'''doc'''

# local imports
import alias
import dataset
import utils

def build_schema(
    dataset_config: alias.ConfigType,
    artifact_config: alias.ConfigType,
):
    '''doc'''

    # config accssors
    dataset_cfg = utils.ConfigAccess(dataset_config)
    artifact_cfg = utils.ConfigAccess(artifact_config)

    #
