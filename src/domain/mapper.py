'''
Tools for preparing and loading domain knowledge.
'''

# local imports
import alias
import domain
import grid
import utils

def prepare_domain(
    grid_id: str,
    world_grid: grid.GridLayout,
    config: alias.ConfigType,
    logger: utils.Logger
) -> dict[str, domain.DomainTileMap]:
    '''doc'''

    # get a child logger
    logger = logger.get_child('dkmap')
    # config accessor
    cfg = utils.ConfigAccess(config)

    source_dir = cfg.get_option('dirpath')
    output_dir = cfg.get_option('output_dirpath')
    output: dict[str, domain.DomainTileMap] = {}
    # prep each domain from configured list
    for file in cfg.get_option('files'):
        name = str(file['name']).split('.', maxsplit=1)[0] # no suffix
        # if domain already exist, load and add to return
        try:
            dom = domain.load_domain(name, output_dir)
            output[name] = dom
        # otherwise create domain accordingly
        except FileNotFoundError:
            ctx = domain.DomainContext(
                index_base=file['index_base'],
                valid_threshold=cfg.get_option('valid_threshold'),
                target_variance=cfg.get_option('target_variance')
            )
            fp = f'{source_dir}/{file["name"]}'
            dom = domain.DomainTileMap(fp, world_grid, ctx, logger)
            output[name] = dom
            domain.save_domain(grid_id, name, dom, output_dir)

    # return
    return output
