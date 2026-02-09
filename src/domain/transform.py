'''doc'''

# third-party imports
import numpy

def pca_transform(
    freqs: dict[tuple[int, int], numpy.ndarray],
    target_var: float
) -> tuple[dict[tuple[int, int], numpy.ndarray], float]:
    '''Perform PCA and output top axes that meets the target variance.'''

    # lock ordering in one pass
    items = list(freqs.items())
    keys = [k for k, _ in items]
    rows = [v for _, v in items]

    # (N, D) stack
    freq_stack = numpy.stack(rows, axis=0)

    # fit once, then choose k
    mean, components_full, evr_full = _fit_pca(freq_stack)
    k = _k_from_target_evr(evr_full, target_var)

    # slice the first k components and explained variance
    components = components_full[:k, :]
    evr_k = evr_full[:k]

    # project
    z = _transform(freq_stack, mean, components)  # (N, k)

    # map z back to freqs keys
    mapped_z = dict(zip(keys, z))
    return mapped_z, float(evr_k.sum() * 100.0)

def _fit_pca(x: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    '''
    Fit PCA on rows of X (N x D) using SVD.
    Returns mean (D,), components_full (D x D_or_N), evr_full (L,).
    '''
    assert x.ndim == 2, x.shape
    # center
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # economical SVD on centered data
    _, s, vt = numpy.linalg.svd(xc, full_matrices=False) # Xc = U S V^T
    # vt shape: (L, D) where L = min(N, D)
    var = (s ** 2) / (x.shape[0] - 1) # per-PC variance (L,)
    total_var = (xc ** 2).sum() / (x.shape[0] - 1)
    evr = var / total_var
    return mean.squeeze(0), vt, evr

def _k_from_target_evr(
    evr_full: numpy.ndarray,
    target: float
) -> int:
    '''
    Return the smallest k s.t. cumulative explained variance >= target.
    target is in [0, 1], e.g., 0.95 for 95%.
    '''

    if not 0.0 < target <= 1.0:
        raise ValueError('target must be in (0, 1].')
    if not numpy.all(numpy.isfinite(evr_full)):
        print("DEBUG: evr_full has non-finite values:", evr_full)
    cum = numpy.cumsum(evr_full)
    print("DEBUG: target=", target, " first_evr=", evr_full[0], " cum_last=", cum[-1])
    k = int(numpy.searchsorted(cum, target, side='left') + 1)
    k = max(1, min(k, evr_full.shape[0]))
    return k

def _transform(
        x: numpy.ndarray,
        mean: numpy.ndarray,
        components: numpy.ndarray
    ) -> numpy.ndarray:
    '''Project rows of X onto PCA components (k x D).'''
    xc = x - mean
    z = xc @ components.T  # (N, k)
    z = numpy.asarray(z, dtype=numpy.float32) # ensure float32
    z = numpy.atleast_2d(z) # ensure 2D shape
    return z
