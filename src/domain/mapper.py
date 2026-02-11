'''
Tools for preparing domain knowledge mapped onto a world grid.

This module provides the public API for constructing per-domain
DomainTileMap objects. It coordinates the following operations:

    - Align the provided world grid to each domain raster using the
        grid's pixel-offset calculation.
    - Load an existing DomainTileMap artifact if present, validating its
        schema and integrity.
    - Otherwise, build a new DomainTileMap by reading all grid windows
        from the categorical raster, filtering tiles by valid-pixel
        fraction, computing majority statistics, deriving normalized
        frequency vectors, and projecting them onto PCA components to
        reach a target explained variance.
    - Persist each new DomainTileMap as a JSON payload and a metadata
        sidecar including schema_id, context, hash, and grid association.

Configuration is supplied via a structured dictionary containing:
    - 'dirpath': directory with domain rasters.
    - 'files': list of domain raster entries ('name', 'index_base').
    - 'valid_threshold': minimum fraction of valid pixels for tile
        acceptance.
    - 'target_variance': PCA target cumulative explained variance.
    - 'output_dirpath': where DomainTileMap artifacts are written.

The output is a mapping from domain base names (filename without suffix)
to DomainTileMap instances, suitable for downstream model conditioning or
task-level feature assembly.
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
    '''
    Prepare and persist domain tile maps for categorical raster(s).

    This function is the public entry point for domain preparation. It
    aligns the provided world grid to each configured domain raster,
    builds a `DomainTileMap` if no artifact exists yet, persists it, and
    returns a dictionary keyed by domain name (filename without suffix).

    Args:
        grid_id: Identifier of the world grid; written to domain metadata
            to preserve traceability between grid and domain artifacts.
        world_grid: A grid.GridLayout instance describing the tiling to
            use. The domain rasters must share its CRS and pixel size;
            pixel origin alignment is handled internally.
        config: Domain configuration dict. Expected keys
            - 'dirpath': directory containing domain rasters.
            - 'files': list of {'name': str, 'index_base': int}.
            - 'valid_threshold': float in [0, 1], min valid-pixel frac.
            - 'target_variance': float in (0, 1], PCA target EVR.
            - 'output_dirpath': directory for persisted artifacts.
        logger:
            Logger used for structured progress messages.

    Returns:
        dict: A mapping from domain base name to the prepared
            `DomainTileMap`.

    Notes:
    - Existing domain artifacts are loaded and returned without rebuild.
    - New artifacts are saved as JSON payload plus JSON metadata with a
    schema id and integrity hash for compatibility checks.
    '''

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
