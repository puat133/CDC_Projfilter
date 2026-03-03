"""
Sample-Based Distribution Comparison Metrics

This module provides efficient JAX implementations of sample-based distance metrics
for comparing probability distributions using samples. These metrics avoid density
estimation and work well in high dimensions.

Metrics Implemented:
1. Maximum Mean Discrepancy (MMD) - Kernel-based metric
2. Energy Distance - Parameter-free metric based on pairwise distances
3. Sliced Wasserstein Distance - Based on 1D projections

All implementations use JAX's vmap, lax, and JIT compilation for optimal performance.

References:
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Sch�lkopf, B., & Smola, A. (2012).
    "A Kernel Two-Sample Test." Journal of Machine Learning Research, 13(1), 723-773.

    Sz�kely, G. J., & Rizzo, M. L. (2013). "Energy Statistics: A Class of Statistics
    Based on Distances." Journal of Statistical Planning and Inference, 143(8), 1249-1272.

    Bonneel, N., Rabin, J., Peyr�, G., & Pfister, H. (2015). "Sliced and Radon
    Wasserstein Barycenters of Measures." Journal of Mathematical Imaging and Vision,
    51(1), 22-45.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import lax, vmap
from functools import partial
from typing import Optional, Tuple
from jaxtyping import Array
from equinox import filter_jit


# ==============================================================================
# Kernel Functions
# ==============================================================================

@filter_jit
def rbf_kernel(X: Array, Y: Array, gamma: float = 1.0) -> Array:
    """
    Compute RBF (Gaussian) kernel matrix between two sample sets.

    K(x, y) = exp(-gamma * ||x - y||�)

    Optimized using JAX vectorization to avoid explicit loops.

    Args:
        X: Array of shape (n, d) - first sample set
        Y: Array of shape (m, d) - second sample set
        gamma: Kernel bandwidth parameter (default: 1.0)

    Returns:
        Kernel matrix of shape (n, m)

    Examples:
        >>> X = jnp.array([[0., 0.], [1., 1.]])
        >>> Y = jnp.array([[0., 0.], [2., 2.]])
        >>> K = rbf_kernel(X, Y, gamma=1.0)
        >>> K.shape
        (2, 2)
    """
    # Compute pairwise squared distances efficiently
    # ||x - y||� = ||x||� + ||y||� - 2�x, y�
    X_sqnorm = jnp.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    Y_sqnorm = jnp.sum(Y**2, axis=1, keepdims=True)  # (m, 1)

    # Pairwise inner products via matrix multiplication
    XY = jnp.dot(X, Y.T)  # (n, m)

    # Squared distances
    sq_dists = X_sqnorm + Y_sqnorm.T - 2 * XY  # (n, m)

    # Clip to avoid numerical issues with negative values from floating point errors
    sq_dists = jnp.maximum(sq_dists, 0.0)

    return jnp.exp(-gamma * sq_dists)


@filter_jit
def compute_median_heuristic_gamma(X: Array, Y: Array,
                                   max_samples: int = 1000) -> float:
    """
    Compute gamma for RBF kernel using median heuristic.

    gamma = 1 / median(pairwise_distances�)

    This is a common heuristic for choosing kernel bandwidth automatically.

    Args:
        X: Samples from distribution P, shape (n, d)
        Y: Samples from distribution Q, shape (m, d)
        max_samples: Maximum number of samples to use for median computation
                    (for efficiency with large datasets)

    Returns:
        gamma: Kernel bandwidth parameter

    References:
        Gretton et al. (2012). Section 3.2: "Choosing the kernel".
    """
    # Subsample for efficiency
    n, m = X.shape[0], Y.shape[0]
    n_subset = min(max_samples // 2, n)
    m_subset = min(max_samples // 2, m)

    X_subset = X[:n_subset]
    Y_subset = Y[:m_subset]

    # Combine samples
    combined = jnp.vstack([X_subset, Y_subset])

    # Compute pairwise squared distances using broadcasting
    # Shape: (n_combined, 1, d) - (1, n_combined, d) -> (n_combined, n_combined, d)
    diffs = combined[:, None, :] - combined[None, :, :]
    sq_dists = jnp.sum(diffs**2, axis=2)  # (n_combined, n_combined)

    # Get upper triangle (exclude diagonal and duplicates)
    # Instead of boolean indexing, compute median over entire matrix
    # but weight the upper triangle values
    # This is JIT-compatible
    n_combined = combined.shape[0]
    i_indices, j_indices = jnp.triu_indices(n_combined, k=1)
    sq_dists_upper = sq_dists[i_indices, j_indices]

    # Compute median
    median_sq_dist = jnp.median(sq_dists_upper)

    # Avoid division by zero
    median_sq_dist = jnp.maximum(median_sq_dist, 1e-8)

    return 1.0 / median_sq_dist


# ==============================================================================
# Maximum Mean Discrepancy (MMD)
# ==============================================================================

@filter_jit
def maximum_mean_discrepancy(X: Array, Y: Array,
                             gamma: Optional[float] = None,
                             use_unbiased: bool = True) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two sample sets.

    MMD�(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]

    where X, X' ~ P, Y, Y' ~ Q, and k is an RBF kernel.

    This is a kernel-based two-sample test that measures the distance between
    two distributions in a reproducing kernel Hilbert space (RKHS).
    The metric has been used in filtering context such as in:


    Optimized using:
    - JAX JIT compilation
    - Vectorized kernel computation
    - Unbiased estimator (removes diagonal terms)

    Args:
        X: Samples from distribution P, shape (n, d)
        Y: Samples from distribution Q, shape (m, d)
        gamma: RBF kernel bandwidth. If None, uses median heuristic.
        use_unbiased: If True, uses unbiased estimator (default: True)

    Returns:
        MMD distance (scalar, non-negative)

    References:
        Gretton et al. (2012). "A Kernel Two-Sample Test." JMLR, 13, 723-773.
        (use in nonlinear filtering)
        Ensemble Transport Filter via Optimized Maximum Mean Discrepancy. https://arxiv.org/abs/2407.11518
        Similarity-based Particle Filter for Remaining Useful Life prediction with enhanced performance https://doi.org/10.1016/j.asoc.2020.106474
        Data-Driven Approximation of Stationary Nonlinear Filters with Optimal Transport Maps.  https://doi.org/10.1109/CDC56724.2024.10886712
    Examples:
        >>> # Two identical distributions should have MMD H 0
        >>> X = jrandom.normal(jrandom.PRNGKey(0), (100, 4))
        >>> Y = jrandom.normal(jrandom.PRNGKey(0), (100, 4))
        >>> mmd = maximum_mean_discrepancy(X, Y)
        >>> mmd < 0.01
        True

        >>> # Different distributions should have MMD > 0
        >>> Y_shifted = Y + 2.0
        >>> mmd = maximum_mean_discrepancy(X, Y_shifted)
        >>> mmd > 0.5
        True
    """
    # Compute gamma using median heuristic if not provided
    if gamma is None:
        gamma = compute_median_heuristic_gamma(X, Y)

    n, m = X.shape[0], Y.shape[0]

    # Compute kernel matrices
    K_XX = rbf_kernel(X, X, gamma)  # (n, n)
    K_YY = rbf_kernel(Y, Y, gamma)  # (m, m)
    K_XY = rbf_kernel(X, Y, gamma)  # (n, m)

    if use_unbiased:
        # Unbiased estimator: remove diagonal terms
        # E[k(X, X')] where X ` X'
        K_XX_sum = jnp.sum(K_XX) - jnp.trace(K_XX)
        K_YY_sum = jnp.sum(K_YY) - jnp.trace(K_YY)

        term1 = K_XX_sum / (n * (n - 1)) if n > 1 else 0.0
        term2 = K_YY_sum / (m * (m - 1)) if m > 1 else 0.0
        term3 = jnp.mean(K_XY)

        mmd_squared = term1 + term2 - 2 * term3
    else:
        # Biased estimator: include diagonal
        mmd_squared = jnp.mean(K_XX) + jnp.mean(K_YY) - 2 * jnp.mean(K_XY)

    # Return MMD (take sqrt, clip to avoid negative values from numerical errors)
    return jnp.sqrt(jnp.maximum(mmd_squared, 0.0))


# ==============================================================================
# Energy Distance
# ==============================================================================

@filter_jit
def pairwise_euclidean_distances(X: Array, Y: Array) -> Array:
    """
    Compute pairwise Euclidean distances between two sample sets.

    Optimized using vectorized operations instead of explicit loops.

    Args:
        X: Array of shape (n, d)
        Y: Array of shape (m, d)

    Returns:
        Distance matrix of shape (n, m)

    Examples:
        >>> X = jnp.array([[0., 0.], [1., 1.]])
        >>> Y = jnp.array([[0., 0.], [3., 4.]])
        >>> D = pairwise_euclidean_distances(X, Y)
        >>> D.shape
        (2, 2)
        >>> jnp.allclose(D[0, 1], 5.0)
        True
    """
    # ||x - y||� = ||x||� + ||y||� - 2�x, y�
    X_sqnorm = jnp.sum(X**2, axis=1, keepdims=True)  # (n, 1)
    Y_sqnorm = jnp.sum(Y**2, axis=1, keepdims=True)  # (m, 1)

    sq_dists = X_sqnorm + Y_sqnorm.T - 2 * jnp.dot(X, Y.T)  # (n, m)

    # Clip to avoid sqrt of negative numbers from numerical errors
    sq_dists = jnp.maximum(sq_dists, 0.0)

    return jnp.sqrt(sq_dists)


@filter_jit
def energy_distance(X: Array, Y: Array) -> float:
    """
    Compute Energy Distance (Cram�r distance) between two sample sets.

    E(P, Q) = 2E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    where X, X' are independent samples from P and Y, Y' are independent from Q.

    This is a metric (satisfies triangle inequality) that requires no tuning
    parameters, unlike kernel methods. It is invariant to rotations and reflections.

    Computational complexity: O(n� + m�) for distance computations

    Optimized using:
    - JAX JIT compilation
    - Vectorized distance computation
    - Efficient sum operations

    Args:
        X: Samples from distribution P, shape (n, d)
        Y: Samples from distribution Q, shape (m, d)

    Returns:
        Energy distance (scalar, non-negative)

    References:
        Sz�kely, G. J., & Rizzo, M. L. (2013). "Energy Statistics: A Class of
        Statistics Based on Distances." Journal of Statistical Planning and
        Inference, 143(8), 1249-1272.

    Examples:
        >>> X = jrandom.normal(jrandom.PRNGKey(0), (50, 4))
        >>> Y = jrandom.normal(jrandom.PRNGKey(0), (50, 4))
        >>> e_dist = energy_distance(X, Y)
        >>> e_dist < 0.1  # Should be small for identical distributions
        True
    """
    n, m = X.shape[0], Y.shape[0]

    # Compute pairwise distances
    XY_dists = pairwise_euclidean_distances(X, Y)  # (n, m)
    XX_dists = pairwise_euclidean_distances(X, X)  # (n, n)
    YY_dists = pairwise_euclidean_distances(Y, Y)  # (m, m)

    # Energy distance formula
    # E[||X - Y||]
    term1 = jnp.sum(XY_dists) / (n * m)

    # E[||X - X'||] (exclude diagonal for unbiased estimate)
    XX_sum = jnp.sum(XX_dists) - jnp.trace(XX_dists)
    term2 = XX_sum / (n * (n - 1)) if n > 1 else 0.0

    # E[||Y - Y'||] (exclude diagonal for unbiased estimate)
    YY_sum = jnp.sum(YY_dists) - jnp.trace(YY_dists)
    term3 = YY_sum / (m * (m - 1)) if m > 1 else 0.0

    return 2 * term1 - term2 - term3


# ==============================================================================
# Sliced Wasserstein Distance
# ==============================================================================

@filter_jit
def wasserstein_1d(x: Array, y: Array) -> float:
    """
    Compute 1D Wasserstein-1 distance between two univariate samples.

    For 1D distributions, the Wasserstein distance has a closed form:
    W�(P, Q) = +|F_P{�(u) - F_Q{�(u)| du

    which equals the L1 distance between sorted samples (quantile functions).

    Handles different sample sizes via linear interpolation.

    Args:
        x: 1D array of samples from distribution P
        y: 1D array of samples from distribution Q

    Returns:
        Wasserstein-1 distance (scalar, non-negative)

    Examples:
        >>> x = jnp.array([1., 2., 3., 4., 5.])
        >>> y = jnp.array([1., 2., 3., 4., 5.])
        >>> wasserstein_1d(x, y)
        Array(0., dtype=float32)
    """
    # Sort both samples (quantile function)
    x_sorted = jnp.sort(x)
    y_sorted = jnp.sort(y)

    n, m = len(x_sorted), len(y_sorted)

    # Handle different sample sizes via interpolation
    def interpolate_to_common_size():
        """Interpolate both to common size via CDF matching."""
        # Interpolate to common grid
        if n < m:
            # Interpolate x to size m
            indices = jnp.linspace(0, n - 1, m)
            x_interp = jnp.interp(indices, jnp.arange(n), x_sorted)
            return x_interp, y_sorted
        elif m < n:
            # Interpolate y to size n
            indices = jnp.linspace(0, m - 1, n)
            y_interp = jnp.interp(indices, jnp.arange(m), y_sorted)
            return x_sorted, y_interp
        else:
            return x_sorted, y_sorted

    x_common, y_common = interpolate_to_common_size()

    # Wasserstein-1 distance is L1 distance between quantile functions
    return jnp.mean(jnp.abs(x_common - y_common))


@filter_jit
def sliced_wasserstein_distance(X: Array, Y: Array,
                                n_projections: int = 100,
                                seed: int = 0) -> float:
    """
    Compute Sliced Wasserstein Distance between two sample sets.

    Projects samples onto random 1D directions and averages the 1D Wasserstein
    distances. This provides a computationally efficient approximation to the
    full multi-dimensional Wasserstein distance.

    SW(P, Q) = E_�[W�(P_�, Q_�)]

    where P_�, Q_� are 1D projections onto random unit vector �  S^(d-1).

    Computational complexity: O(n_proj * n log n) much faster than full
    Wasserstein which is O(n� log n).

    Optimized using:
    - JAX lax.scan for efficient iteration
    - Vectorized projections
    - JIT compilation

    Args:
        X: Samples from distribution P, shape (n, d)
        Y: Samples from distribution Q, shape (m, d)
        n_projections: Number of random 1D projections (default: 100)
        seed: Random seed for reproducibility

    Returns:
        Sliced Wasserstein distance (scalar, non-negative)

    References:
        Bonneel et al. (2015). "Sliced and Radon Wasserstein Barycenters of
        Measures." Journal of Mathematical Imaging and Vision, 51(1), 22-45.
        Kolouri et al. (2019). "Generalized Sliced Wasserstein Distances." NeurIPS.
        (use in nonlinear filtering)
        APPROXIMATE BAYESIAN COMPUTATION WITH THE SLICED-WASSERSTEIN DISTANCE, https://arxiv.org/pdf/1910.12815

    Examples:
        >>> X = jrandom.normal(jrandom.PRNGKey(0), (100, 4))
        >>> Y = jrandom.normal(jrandom.PRNGKey(1), (100, 4))
        >>> sw = sliced_wasserstein_distance(X, Y, n_projections=50)
        >>> sw > 0
        True
    """
    d = X.shape[1]
    key = jrandom.PRNGKey(seed)

    def compute_single_projection(carry, _):
        """Compute Wasserstein distance for one random projection."""
        rng_key = carry

        # Generate random unit vector
        rng_key, subkey = jrandom.split(rng_key)
        theta = jrandom.normal(subkey, shape=(d,))
        theta = theta / jnp.linalg.norm(theta)

        # Project samples onto theta
        X_proj = jnp.dot(X, theta)  # (n,)
        Y_proj = jnp.dot(Y, theta)  # (m,)

        # Compute 1D Wasserstein distance
        w_dist = wasserstein_1d(X_proj, Y_proj)

        return rng_key, w_dist

    # Use lax.scan for efficient iteration (avoids Python loop overhead)
    _, distances = lax.scan(
        compute_single_projection,
        key,
        None,
        length=n_projections
    )

    # Average over all projections
    return jnp.mean(distances)


@filter_jit
def sliced_wasserstein_distance_parallel(X: Array, Y: Array,
                                          n_projections: int = 100,
                                          seed: int = 0) -> float:
    """
    Compute Sliced Wasserstein Distance with parallel projection computation.

    OPTIMIZATION: Uses vmap instead of lax.scan for potentially faster parallel
    execution on multi-core CPUs and GPUs. This trades slight memory overhead
    for better parallelization.

    Projects samples onto random 1D directions and averages the 1D Wasserstein
    distances.

    Args:
        X: Samples from distribution P, shape (n, d)
        Y: Samples from distribution Q, shape (m, d)
        n_projections: Number of random 1D projections (default: 100)
        seed: Random seed for reproducibility

    Returns:
        Sliced Wasserstein distance (scalar, non-negative)
    """
    d = X.shape[1]

    # Generate all random keys at once
    keys = jrandom.split(jrandom.PRNGKey(seed), n_projections)

    def compute_projection_distance(key):
        """Compute Wasserstein distance for one random projection."""
        # Generate random unit vector
        theta = jrandom.normal(key, shape=(d,))
        theta = theta / jnp.linalg.norm(theta)

        # Project samples onto theta
        X_proj = jnp.dot(X, theta)  # (n,)
        Y_proj = jnp.dot(Y, theta)  # (m,)

        # Compute 1D Wasserstein distance
        return wasserstein_1d(X_proj, Y_proj)

    # Use vmap for parallel computation across projections
    distances = jax.vmap(compute_projection_distance)(keys)

    # Average over all projections
    return jnp.mean(distances)


# ==============================================================================
# Batch Computation Functions
# ==============================================================================

@filter_jit
def compute_metric_batch(X_batch: Array, Y_batch: Array,
                        metric_fn, **kwargs) -> Array:
    """
    Compute a distance metric for batches of sample pairs.

    Useful for computing metrics across multiple timesteps or multiple
    sample pairs in parallel.

    Args:
        X_batch: Batch of samples from P, shape (batch_size, n, d)
        Y_batch: Batch of samples from Q, shape (batch_size, m, d)
        metric_fn: Distance metric function (mmd, energy_distance, etc.)
        **kwargs: Additional arguments for metric_fn

    Returns:
        Array of distances, shape (batch_size,)

    Examples:
        >>> X_batch = jrandom.normal(jrandom.PRNGKey(0), (10, 50, 4))
        >>> Y_batch = jrandom.normal(jrandom.PRNGKey(1), (10, 50, 4))
        >>> distances = compute_metric_batch(X_batch, Y_batch, energy_distance)
        >>> distances.shape
        (10,)
    """
    # Use vmap to vectorize over the batch dimension
    batched_metric = vmap(lambda x, y: metric_fn(x, y, **kwargs))
    return batched_metric(X_batch, Y_batch)


# ==============================================================================
# Utility Functions
# ==============================================================================

def compute_all_metrics(X: Array, Y: Array,
                       gamma: Optional[float] = None,
                       n_projections: int = 100,
                       seed: int = 0) -> dict:
    """
    Compute all three sample-based distance metrics.

    Convenience function to compute MMD, Energy Distance, and Sliced Wasserstein
    in one call.

    Args:
        X: Samples from distribution P, shape (n, d)
        Y: Samples from distribution Q, shape (m, d)
        gamma: RBF kernel bandwidth for MMD (None = median heuristic)
        n_projections: Number of projections for Sliced Wasserstein
        seed: Random seed for Sliced Wasserstein

    Returns:
        Dictionary with keys:
            'mmd': Maximum Mean Discrepancy
            'energy': Energy Distance
            'sliced_wasserstein': Sliced Wasserstein Distance

    Examples:
        >>> X = jrandom.normal(jrandom.PRNGKey(0), (100, 4))
        >>> Y = jrandom.normal(jrandom.PRNGKey(1), (100, 4))
        >>> metrics = compute_all_metrics(X, Y)
        >>> 'mmd' in metrics and 'energy' in metrics
        True
    """
    return {
        'mmd': maximum_mean_discrepancy(X, Y, gamma=gamma),
        'energy': energy_distance(X, Y),
        'sliced_wasserstein': sliced_wasserstein_distance(X, Y, n_projections, seed)
    }


# ==============================================================================
# Subsample Function for Large Datasets
# ==============================================================================

@filter_jit
def subsample_arrays(X: Array, Y: Array, max_samples: int,
                    key: jrandom.PRNGKey) -> Tuple[Array, Array]:
    """
    Randomly subsample arrays for computational efficiency.

    Useful when working with very large sample sets where computing pairwise
    distances becomes prohibitive.

    Args:
        X: Samples from P, shape (n, d)
        Y: Samples from Q, shape (m, d)
        max_samples: Maximum samples to keep from each array
        key: JAX random key

    Returns:
        Tuple of (X_sub, Y_sub) with at most max_samples each

    Examples:
        >>> X = jrandom.normal(jrandom.PRNGKey(0), (10000, 4))
        >>> Y = jrandom.normal(jrandom.PRNGKey(1), (10000, 4))
        >>> key = jrandom.PRNGKey(42)
        >>> X_sub, Y_sub = subsample_arrays(X, Y, 1000, key)
        >>> X_sub.shape[0] <= 1000 and Y_sub.shape[0] <= 1000
        True
    """
    n, m = X.shape[0], Y.shape[0]

    key_x, key_y = jrandom.split(key)

    # Subsample X if needed
    if n > max_samples:
        indices_x = jrandom.choice(key_x, n, shape=(max_samples,), replace=False)
        X_sub = X[indices_x]
    else:
        X_sub = X

    # Subsample Y if needed
    if m > max_samples:
        indices_y = jrandom.choice(key_y, m, shape=(max_samples,), replace=False)
        Y_sub = Y[indices_y]
    else:
        Y_sub = Y

    return X_sub, Y_sub
