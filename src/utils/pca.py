'''PCA utility.'''

# third-party imports
import numpy

def pca_transform(x, k) -> tuple[numpy.ndarray, float]:
    '''doc'''

    mean, components, evr = _fit_pca(x, k)
    z = _transform(x, mean, components)  # (N, k)

    return z, float(evr.sum())

def _fit_pca(
        x: numpy.ndarray,
        k: int
    ) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    '''
    Fit PCA on rows of X (N x D) using SVD.
    Returns mean (D,), components (k x D), explained_variance_ratio (k,).
    '''
    assert x.ndim == 2, x.shape
    # center
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    # economical SVD on centered data
    _, s, vt = numpy.linalg.svd(xc, full_matrices=False) # Xc = U S V^T
    components = vt[:k, :] # (k, D)
    # explained variance ratio
    var = (s ** 2) / (x.shape[0] - 1) # per-PC variance
    total_var = (xc ** 2).sum() / (x.shape[0] - 1)
    evr = var[:k] / total_var
    return mean.squeeze(0), components, evr

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
