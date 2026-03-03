"""
Fast EM (Expectation-Maximization) algorithm optimized for JAX for CD-PGM filter.

This module provides a lightweight, JIT-compiled EM implementation
specifically optimized for the CD-PGM filter use case, where:
- We re-fit GMM at every measurement step
- We need maximum performance for large sample sizes (100K-400K samples)
- Output should be directly usable as GMM parameters (means, covs, weights)

The implementation follows OTT's fit_gmm but is optimized for:
- Pure JAX operations for full JIT compilation
- Purely functional approach (no classes)
- Direct output of (means, covs, weights) tuple
- K-means++ initialization (same as OTT)

Default parameters:
- max_iterations: required parameter (maximum number of iterations)
- tol: 1e-4 (convergence tolerance based on log-likelihood change)
- reg=0.0 (no regularization by default, same as OTT)

The implementation uses:
- Pure JAX operations for full JIT compilation
- Log-space computations for numerical stability (log-sum-exp)
- Explicit @filter_jit and jax.vmap for all functions
- Purely functional approach (no classes)
- while_loop for convergence-based iteration

Output format (tuple):
    (means, covs, weights)
    - means: jax.Array, shape (k, ndim) - component means
    - covs: jax.Array, shape (k, ndim, ndim) - component covariances
    - weights: jax.Array, shape (k,) - component weights (mixture proportions)
"""
from typing import Optional, Tuple

import jax
from equinox import filter_jit
import jax.numpy as jnp
import jax.random as jrandom
from jax.lax import fori_loop, while_loop
from jaxtyping import Array
import math

LOG2PI = math.log(2.0 * math.pi)


@filter_jit
def compute_log_det_cholesky(cholesky: Array) -> Array:
    """
    Compute log determinant of covariance from its Cholesky factor.

    Parameters
    ----------
    cholesky : Array
        Cholesky factor L where cov = L @ L.T, shape (ndim, ndim)

    Returns
    -------
    Array
        Log determinant of the covariance matrix (scalar)
    """
    return 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))


@filter_jit
def compute_log_prob_single_component(
    points: Array,
    mean: Array,
    cholesky: Array,
    log_det: Array,
) -> Array:
    """
    Compute log probability of points under a single Gaussian component.

    Uses Cholesky decomposition for numerical stability.

    Parameters
    ----------
    points : Array
        Points array, shape (n, ndim)
    mean : Array
        Component mean, shape (ndim,)
    cholesky : Array
        Cholesky factor of covariance, shape (ndim, ndim)
    log_det : Array
        Log determinant of covariance (scalar)

    Returns
    -------
    Array
        Log probabilities, shape (n,)
    """
    ndim = points.shape[1]
    centered = points - mean  # (n, ndim)
    # Solve L @ z = centered.T for z, then ||z||^2 = (centered @ cov^{-1} @ centered.T)
    # Using solve_triangular for numerical stability
    z = jax.scipy.linalg.solve_triangular(
        cholesky, centered.T, lower=True
    ).T  # (n, ndim)
    mahalanobis_sq = jnp.sum(z ** 2, axis=1)  # (n,)
    return -0.5 * (ndim * LOG2PI + log_det + mahalanobis_sq)


@filter_jit
def compute_conditional_log_prob(
    points: Array,
    means: Array,
    choleskys: Array,
    log_dets: Array,
) -> Array:
    """
    Compute log p(x | component) for all components.

    Parameters
    ----------
    points : Array
        Points array, shape (n, ndim)
    means : Array
        Component means, shape (k, ndim)
    choleskys : Array
        Cholesky factors of covariances, shape (k, ndim, ndim)
    log_dets : Array
        Log determinants of covariances, shape (k,)

    Returns
    -------
    Array
        Conditional log probabilities, shape (n, k)
    """
    def single_component_log_prob(mean, cholesky, log_det):
        return compute_log_prob_single_component(points, mean, cholesky, log_det)

    return jax.vmap(single_component_log_prob, in_axes=(0, 0, 0), out_axes=1)(
        means, choleskys, log_dets
    )


@filter_jit
def e_step(
    points: Array,
    means: Array,
    choleskys: Array,
    log_dets: Array,
    log_weights: Array,
) -> Tuple[Array, Array]:
    """
    E-step: compute posterior assignment probabilities.

    Computes p(component | x) = p(x | component) * p(component) / p(x)

    Parameters
    ----------
    points : Array
        Points array, shape (n, ndim)
    means : Array
        Component means, shape (k, ndim)
    choleskys : Array
        Cholesky factors of covariances, shape (k, ndim, ndim)
    log_dets : Array
        Log determinants of covariances, shape (k,)
    log_weights : Array
        Log component weights, shape (k,)

    Returns
    -------
    Tuple[Array, Array]
        assignment_probs : Posterior probabilities, shape (n, k)
        log_likelihood : Total log-likelihood (scalar)
    """
    # log p(x | component) for all components: (n, k)
    log_prob_cond = compute_conditional_log_prob(points, means, choleskys, log_dets)

    # log p(x, component) = log p(x | component) + log p(component)
    log_joint = log_prob_cond + log_weights[None, :]  # (n, k)

    # log p(x) = logsumexp over components
    log_prob_x = jax.scipy.special.logsumexp(log_joint, axis=1, keepdims=True)  # (n, 1)

    # log p(component | x) = log p(x, component) - log p(x)
    log_posterior = log_joint - log_prob_x  # (n, k)

    # Convert to probabilities
    assignment_probs = jnp.exp(log_posterior)

    # Total log-likelihood
    log_likelihood = jnp.mean(log_prob_x)

    return assignment_probs, log_likelihood


@filter_jit
def m_step(
    points: Array,
    assignment_probs: Array,
    reg: float = 1e-6,
) -> Tuple[Array, Array, Array]:
    """
    M-step: update GMM parameters given assignment probabilities.

    Parameters
    ----------
    points : Array
        Points array, shape (n, ndim)
    assignment_probs : Array
        Posterior assignment probabilities, shape (n, k)
    reg : float
        Regularization added to covariance diagonal for numerical stability.

    Returns
    -------
    Tuple[Array, Array, Array]
        means : Updated component means, shape (k, ndim)
        covs : Updated component covariances, shape (k, ndim, ndim)
        weights : Updated component weights, shape (k,)
    """
    n_samples, ndim = points.shape
    k = assignment_probs.shape[1]

    # Component weights: sum of responsibilities normalized
    weights_sum = jnp.sum(assignment_probs, axis=0)  # (k,)
    weights = weights_sum / n_samples  # (k,)

    def compute_single_component_params(j: int) -> Tuple[Array, Array]:
        """Compute mean and covariance for component j."""
        resp = assignment_probs[:, j]  # (n,)
        resp_sum = weights_sum[j]

        # Weighted mean
        weighted_points = resp[:, None] * points  # (n, ndim)
        mean = jnp.sum(weighted_points, axis=0) / jnp.maximum(resp_sum, 1e-10)  # (ndim,)

        # Weighted covariance
        centered = points - mean  # (n, ndim)
        # cov = sum(resp * outer(centered, centered)) / sum(resp)
        outer_products = jnp.einsum('ni,nj->nij', centered, centered)  # (n, ndim, ndim)
        weighted_outer = resp[:, None, None] * outer_products  # (n, ndim, ndim)
        cov = jnp.sum(weighted_outer, axis=0) / jnp.maximum(resp_sum, 1e-10)  # (ndim, ndim)

        # Add regularization
        cov = cov + reg * jnp.eye(ndim)

        return mean, cov

    # Use vmap to compute all component parameters in parallel
    means, covs = jax.vmap(compute_single_component_params)(jnp.arange(k))

    return means, covs, weights


@filter_jit
def compute_cholesky_and_logdet(covs: Array) -> Tuple[Array, Array]:
    """
    Compute Cholesky decompositions and log determinants of covariance matrices.

    Parameters
    ----------
    covs : Array
        Covariance matrices, shape (k, ndim, ndim)

    Returns
    -------
    Tuple[Array, Array]
        choleskys : Cholesky factors, shape (k, ndim, ndim)
        log_dets : Log determinants, shape (k,)
    """
    def single_cholesky_logdet(cov):
        cholesky = jnp.linalg.cholesky(cov)
        log_det = compute_log_det_cholesky(cholesky)
        return cholesky, log_det

    return jax.vmap(single_cholesky_logdet)(covs)


@filter_jit
def kmeans_plusplus_init(points: Array, k: int, key: Array) -> Array:
    """
    K-means++ initialization for selecting initial means.

    Parameters
    ----------
    points : Array
        Points array, shape (n, ndim)
    k : int
        Number of components
    key : Array
        JAX random key

    Returns
    -------
    Array
        Initial means, shape (k, ndim)
    """
    n, ndim = points.shape

    # Select first centroid uniformly at random
    key, subkey = jrandom.split(key)
    first_idx = jrandom.randint(subkey, (), 0, n)
    first_mean = points[first_idx]

    # Initialize means array
    means = jnp.zeros((k, ndim))
    means = means.at[0].set(first_mean)

    # Initialize min distances
    min_distances_sq = jnp.sum((points - first_mean) ** 2, axis=1)

    def body_fn(i: int, state: Tuple) -> Tuple:
        """Select next mean with probability proportional to D^2."""
        means, min_distances_sq, key = state

        key, subkey = jrandom.split(key)
        probs = min_distances_sq / jnp.sum(min_distances_sq)
        # Gumbel-max trick for categorical sampling
        gumbel = jrandom.gumbel(subkey, shape=(n,))
        next_idx = jnp.argmax(jnp.log(probs + 1e-10) + gumbel)

        new_mean = points[next_idx]
        means = means.at[i].set(new_mean)

        # Update min distances
        new_distances_sq = jnp.sum((points - new_mean) ** 2, axis=1)
        min_distances_sq = jnp.minimum(min_distances_sq, new_distances_sq)

        return means, min_distances_sq, key

    means, _, _ = fori_loop(1, k, body_fn, (means, min_distances_sq, key))
    return means


@filter_jit
def initialize_gmm_from_means(
    points: Array,
    means: Array,
    reg: float = 1e-6,
) -> Tuple[Array, Array, Array]:
    """
    Initialize GMM parameters from given means using hard assignment.

    Parameters
    ----------
    points : Array
        Points array, shape (n, ndim)
    means : Array
        Initial means, shape (k, ndim)
    reg : float
        Regularization for covariances.

    Returns
    -------
    Tuple[Array, Array, Array]
        means : Component means, shape (k, ndim)
        covs : Component covariances, shape (k, ndim, ndim)
        weights : Component weights, shape (k,)
    """
    n, ndim = points.shape
    k = means.shape[0]

    # Hard assignment based on distances
    # ||x - mean||^2 for all points and all means
    points_sq = jnp.sum(points ** 2, axis=1, keepdims=True)  # (n, 1)
    means_sq = jnp.sum(means ** 2, axis=1)  # (k,)
    cross = points @ means.T  # (n, k)
    distances_sq = points_sq + means_sq - 2 * cross  # (n, k)

    # Hard assignments
    assignments = jnp.argmin(distances_sq, axis=1)  # (n,)

    # Convert to one-hot assignment probabilities
    assignment_probs = jax.nn.one_hot(assignments, k)  # (n, k)

    # Use M-step to compute parameters
    means_new, covs, weights = m_step(points, assignment_probs, reg)

    return means_new, covs, weights


@filter_jit
def em(
    samples: Array,
    max_iterations: int,
    means_init: Optional[Array] = None,
    covs_init: Optional[Array] = None,
    weights_init: Optional[Array] = None,
    tol: float = 1e-4,
    reg: float = 0.0,
) -> Tuple[Array, Array, Array]:
    """
    EM algorithm for Gaussian Mixture Model fitting.

    This is a fast, JIT-compiled implementation optimized for the CD-PGM filter.
    Returns GMM parameters directly usable for filtering.

    Runs EM iterations until convergence (log-likelihood change < tol) or
    max_iterations is reached.

    Parameters
    ----------
    samples : Array
        Point cloud of shape (n, ndim) to fit.
    max_iterations : int
        Maximum number of EM iterations to perform.
    means_init : Optional[Array]
        Initial component means, shape (k, ndim). Required.
    covs_init : Optional[Array]
        Initial component covariances, shape (k, ndim, ndim). Required.
    weights_init : Optional[Array]
        Initial component weights, shape (k,). Required.
    tol : float
        Convergence tolerance. Iteration stops when |ll_new - ll_old| < tol.
        Default is 1e-4.
    reg : float
        Regularization for covariance matrices. Default is 0.0.

    Returns
    -------
    Tuple[Array, Array, Array]
        means : Component means, shape (k, ndim)
        covs : Component covariances, shape (k, ndim, ndim)
        weights : Component weights (mixture proportions), shape (k,)

    Notes
    -----
    This implementation:
    - Uses log-space computations for numerical stability
    - Is optimized for speed in filtering applications
    - Returns GMM-ready parameters with regularized covariances
    - Uses while_loop for convergence-based iteration

    Performance: Faster than OTT fit_gmm due to full JIT compilation
    and functional implementation.
    """
    # Use provided initial parameters
    means = means_init
    covs = covs_init + reg * jnp.eye(covs_init.shape[-1])
    weights = weights_init

    # Compute initial Cholesky and log-determinants
    choleskys, log_dets = compute_cholesky_and_logdet(covs)
    log_weights = jnp.log(weights + 1e-10)

    # Compute initial log-likelihood
    _, init_ll = e_step(samples, means, choleskys, log_dets, log_weights)

    # EM loop using while_loop for convergence-based iteration
    def cond_fn(state: Tuple) -> bool:
        """Check if we should continue iterating."""
        _, _, _, _, _, _, prev_ll, curr_ll, iteration = state
        not_converged = jnp.abs(curr_ll - prev_ll) >= tol
        not_max_iter = iteration < max_iterations
        return not_converged & not_max_iter

    def body_fn(state: Tuple) -> Tuple:
        """One iteration of EM."""
        means, covs, weights, choleskys, log_dets, log_weights, _, curr_ll, iteration = state

        # E-step
        assignment_probs, _ = e_step(samples, means, choleskys, log_dets, log_weights)

        # M-step
        means_new, covs_new, weights_new = m_step(samples, assignment_probs, reg)

        # Update Cholesky and log-determinants
        choleskys_new, log_dets_new = compute_cholesky_and_logdet(covs_new)
        log_weights_new = jnp.log(weights_new + 1e-10)

        # Compute new log-likelihood
        _, new_ll = e_step(samples, means_new, choleskys_new, log_dets_new, log_weights_new)

        return (
            means_new, covs_new, weights_new,
            choleskys_new, log_dets_new, log_weights_new,
            curr_ll, new_ll, iteration + 1,
        )

    # Initialize with prev_ll = -inf to ensure at least one iteration
    init_state = (
        means, covs, weights,
        choleskys, log_dets, log_weights,
        jnp.array(-jnp.inf), init_ll, 0,
    )
    final_state = while_loop(cond_fn, body_fn, init_state)
    means, covs, weights = final_state[0], final_state[1], final_state[2]

    return means, covs, weights


@filter_jit
def em_with_likelihood(
    samples: Array,
    max_iterations: int,
    means_init: Array,
    covs_init: Array,
    weights_init: Array,
    tol: float = 1e-4,
    reg: float = 0.0,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    EM algorithm returning GMM parameters plus final log-likelihood and iteration count.

    Same as em() but also returns the final log-likelihood and number of iterations.

    Parameters
    ----------
    samples : Array
        Point cloud of shape (n, ndim) to fit.
    max_iterations : int
        Maximum number of EM iterations to perform.
    means_init : Array
        Initial component means, shape (k, ndim).
    covs_init : Array
        Initial component covariances, shape (k, ndim, ndim).
    weights_init : Array
        Initial component weights, shape (k,).
    tol : float
        Convergence tolerance. Iteration stops when |ll_new - ll_old| < tol.
        Default is 1e-4.
    reg : float
        Regularization for covariances. Default is 0.0.

    Returns
    -------
    Tuple[Array, Array, Array, Array, Array]
        means : Component means, shape (k, ndim)
        covs : Component covariances, shape (k, ndim, ndim)
        weights : Component weights, shape (k,)
        log_likelihood : Final average log-likelihood (scalar)
        n_iterations : Number of iterations performed (scalar)
    """
    # Use provided initial parameters
    means = means_init
    covs = covs_init + reg * jnp.eye(covs_init.shape[-1])
    weights = weights_init

    choleskys, log_dets = compute_cholesky_and_logdet(covs)
    log_weights = jnp.log(weights + 1e-10)

    # Compute initial log-likelihood
    _, init_ll = e_step(samples, means, choleskys, log_dets, log_weights)

    # EM loop using while_loop for convergence-based iteration
    def cond_fn(state: Tuple) -> bool:
        """Check if we should continue iterating."""
        _, _, _, _, _, _, prev_ll, curr_ll, iteration = state
        not_converged = jnp.abs(curr_ll - prev_ll) >= tol
        not_max_iter = iteration < max_iterations
        return not_converged & not_max_iter

    def body_fn(state: Tuple) -> Tuple:
        """One iteration of EM."""
        means, covs, weights, choleskys, log_dets, log_weights, _, curr_ll, iteration = state

        # E-step
        assignment_probs, _ = e_step(samples, means, choleskys, log_dets, log_weights)

        # M-step
        means_new, covs_new, weights_new = m_step(samples, assignment_probs, reg)

        # Update Cholesky and log-determinants
        choleskys_new, log_dets_new = compute_cholesky_and_logdet(covs_new)
        log_weights_new = jnp.log(weights_new + 1e-10)

        # Compute new log-likelihood
        _, new_ll = e_step(samples, means_new, choleskys_new, log_dets_new, log_weights_new)

        return (
            means_new, covs_new, weights_new,
            choleskys_new, log_dets_new, log_weights_new,
            curr_ll, new_ll, iteration + 1,
        )

    # Initialize with prev_ll = -inf to ensure at least one iteration
    init_state = (
        means, covs, weights,
        choleskys, log_dets, log_weights,
        jnp.array(-jnp.inf), init_ll, 0,
    )
    final_state = while_loop(cond_fn, body_fn, init_state)
    means, covs, weights = final_state[0], final_state[1], final_state[2]
    final_ll = final_state[7]
    n_iterations = final_state[8]

    return means, covs, weights, final_ll, n_iterations
