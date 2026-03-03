"""
Optimized CD-PGM Filter using custom fast k-means implementation.

This module provides an optimized version of the CD-PGM (Continuous-Discrete
Particle Gaussian Mixture) filter that uses a custom fast k-means implementation
instead of OTT's k-means, providing ~40x speedup.

Key optimization: Replaces OTT k-means (n_init=10, max_iterations=300) with
custom fast_kmeans (single run, max_iterations=50) that directly outputs
GMM parameters (means, covs, weights).
"""
from typing import Callable, Literal

import jax
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
import ott.tools.gaussian_mixture.gaussian_mixture as gm
from diffrax import AbstractSolver
from equinox import filter_jit
from jax.lax import scan
from sigma_points.sigma_point_filter_routines import outer
from utils.diffrax import VectorizedControlTerm
from jaxtyping import PyTree, Array, Float

# Import custom fast k-means
from other_filter.fast_kmeans import k_means_with_assignments


@filter_jit
def cd_pgm_optimized(
    process_drift: Callable[[Float, Array, PyTree], Array],
    process_diffusion: Callable[[Float, Array, PyTree], Array],
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    time_sample: Float,
    means_init: Array,
    covs_init: Array,
    n_samples: int,
    weights_init: Array,
    meas_record: Array,
    process_brownian_dim: int,
    prng_key: Array,
    sde_solver: AbstractSolver = dfx.EulerHeun(),
    dt: Float = 1e-3,
    kmeans_max_iterations: int = 50,
    kmeans_init: Literal['k-means++', 'random'] = 'k-means++',
    kmeans_tol: float = 1e-4,
):
    """
    Optimized CD-PGM Filter using custom fast k-means.

    This is an optimized version of cd_pgm that uses a custom fast k-means
    implementation instead of OTT's k-means, providing ~40x speedup.

    Parameters
    ----------
    process_drift : Callable[[Float, Array, PyTree], Array]
        Function defining the drift term of the process SDE
    process_diffusion : Callable[[Float, Array, PyTree], Array]
        Function defining the diffusion term of the process SDE
    measurement_function : Callable[[Array], Array]
        Function that maps state to measurement space
    measurement_covariance : Array
        Covariance matrix of measurement noise
    time_sample : Float
        Time interval between measurements
    means_init : Array
        Initial means for Gaussian mixture components
    covs_init : Array
        Initial covariance matrices for Gaussian mixture components
    n_samples : int
        Number of particles to use in filter
    weights_init : Array
        Initial weights for Gaussian mixture components
    meas_record : Array
        Array containing measurement sequence
    process_brownian_dim : int
        Dimension of Brownian motion driving the process
    prng_key : Array
        JAX random number generator key
    sde_solver : AbstractSolver, optional
        Solver for SDE simulation, by default dfx.EulerHeun()
    dt : Float, optional
        Time step for SDE solver, by default 1e-3
    kmeans_max_iterations : int, optional
        Maximum iterations for k-means Lloyd algorithm. Default is 50.
    kmeans_init : Literal['k-means++', 'random'], optional
        K-means initialization method. Default is 'k-means++'.
    kmeans_tol : float, optional
        Convergence tolerance for k-means. Default is 1e-4.

    Returns
    -------
    tuple
        means_hist : Array of component means over time
        covs_hist : Array of component covariances over time
        weights_hist : Array of component weights over time
        samples_sol_hist : diffeqsolve solution history (for stats)

    Notes
    -----
    Continuous-Discrete Particle Gauss Mixture Filter, based on:
    Algorithm I in Particle Gaussian mixture filters-I
    DOI 10.1016/j.automatica.2018.07.023

    Performance Comparison (200K samples, 50 clusters):
    - OTT k-means (default): ~272 seconds per call
    - Custom fast k-means: ~7 seconds per call
    - Speedup: ~40x
    """

    cluster_dim = means_init.shape[0]

    def kalman_update_masked(_samples: Array,
                             _mean: Array,
                             _cov: Array,
                             _meas: Array,
                             _mask: Array):
        """
        Kalman update using masked samples for a specific cluster.
        """
        cluster_count = jnp.sum(_mask)

        # Compute deviations using masked samples
        _x_deviation = _samples - _mean  # (n_samples, state_dim)
        _h_samples = measurement_function(_samples)  # (n_samples, meas_dim)

        # Compute masked mean of h
        _h_sum = jnp.sum(jnp.where(_mask[:, None], _h_samples, 0.0), axis=0)
        _h_mean = _h_sum / jnp.maximum(cluster_count, 1)

        _h_deviation = _h_samples - _h_mean  # (n_samples, meas_dim)

        # Compute cross-covariances using masked outer products
        outer_yx = outer(_h_deviation, _x_deviation)
        masked_outer_yx = jnp.where(_mask[:, None, None], outer_yx, 0.0)
        P_yx = jnp.sum(masked_outer_yx, axis=0) / jnp.maximum(cluster_count, 1)

        outer_yy = outer(_h_deviation, _h_deviation)
        masked_outer_yy = jnp.where(_mask[:, None, None], outer_yy, 0.0)
        P_yy = jnp.sum(masked_outer_yy, axis=0) / jnp.maximum(cluster_count, 1) + measurement_covariance

        K = (jnp.linalg.solve(P_yy, P_yx)).T
        _mean = _mean + K @ (_meas - _h_mean)
        _cov = _cov - K @ P_yx

        return _mean, _cov

    def apply_kalman_update_to_cluster(j: int, samples: Array, assignments: Array,
                                       mean: Array, cov: Array, meas: Array):
        """
        Apply Kalman update to a single cluster.
        """
        mask = (assignments == j)
        mean_updated, cov_updated = kalman_update_masked(samples, mean, cov, meas, mask)
        return mean_updated, cov_updated

    def scanned_fun(_carry, _input):
        _prng_key, _means, _covs, _weights, _t = _carry
        _meas = _input

        # Sample from GMM
        _prng_key, subkey = jrandom.split(_prng_key)
        _gmm = gm.GaussianMixture.from_mean_cov_component_weights(_means, _covs, _weights)
        samples = _gmm.sample(subkey, n_samples)

        # Propagate samples through SDE
        _prng_key, subkey = jrandom.split(_prng_key)
        brownian_motion = dfx.UnsafeBrownianPath(
            shape=(n_samples, process_brownian_dim,),
            key=subkey,
            levy_area=dfx.BrownianIncrement
        )
        sde_terms = dfx.MultiTerm(
            dfx.ODETerm(process_drift),
            VectorizedControlTerm(process_diffusion, brownian_motion)
        )

        samples_sol = dfx.diffeqsolve(
            sde_terms,
            sde_solver,
            _t,
            _t + time_sample,
            dt0=dt,
            y0=samples,
            saveat=dfx.SaveAt(t1=True),
            adjoint=dfx.DirectAdjoint()
        )

        samples = samples_sol.ys[-1]

        # K-means clustering using custom fast implementation
        # This directly returns (means, covs, weights, assignments)
        _prng_key, subkey = jrandom.split(_prng_key)
        _means, _covs, _weights, assignments = k_means_with_assignments(
            samples,
            cluster_dim,
            rng=subkey,
            init=kmeans_init,
            max_iterations=kmeans_max_iterations,
            tol=kmeans_tol,
        )

        # Apply Kalman update to each cluster using vmap
        def update_cluster(j, mean, cov):
            return apply_kalman_update_to_cluster(j, samples, assignments, mean, cov, _meas)

        cluster_indices = jnp.arange(cluster_dim)
        _means, _covs = jax.vmap(update_cluster)(cluster_indices, _means, _covs)

        _carry = (_prng_key, _means, _covs, _weights, _t + time_sample)
        return _carry, (_means, _covs, _weights, samples_sol)

    init_carry = (prng_key, means_init, covs_init, weights_init, 0.0)
    _, results = scan(scanned_fun, init_carry, xs=meas_record)

    return results


def create_pgm_drift_for_cost_analysis(
    process_drift: Callable[[Float, Array, PyTree], Array],
) -> Callable:
    """
    Create a standalone JIT-compiled PGM drift function for cost analysis.

    Parameters
    ----------
    process_drift : Callable[[Float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE.

    Returns
    -------
    Callable
        JIT-compiled drift function with signature (t, samples, args) -> d_samples
    """
    @filter_jit
    def drift_derivative(t, samples, args=None):
        return process_drift(t, samples, args)

    return drift_derivative


def create_pgm_update_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    cluster_dim: int,
    kmeans_max_iterations: int = 50,
    kmeans_tol: float = 1e-4,
) -> Callable:
    """
    Create a standalone JIT-compiled PGM update function for cost analysis.

    This measures the full cost of the PGM update step including:
    1. K-means clustering to get GMM parameters and assignments
    2. Kalman update for all clusters

    Parameters
    ----------
    measurement_function : Callable[[Array], Array]
        Function that maps state to measurement space.
    measurement_covariance : Array
        Covariance matrix of measurement noise.
    cluster_dim : int
        Number of Gaussian mixture components.
    kmeans_max_iterations : int
        Maximum iterations for k-means. Default is 50.
    kmeans_tol : float
        Convergence tolerance for k-means. Default is 1e-4.

    Returns
    -------
    Callable
        JIT-compiled update function with signature:
        (samples, meas, prng_key) -> (means, covs, weights)
    """
    def kalman_update_masked(_samples: Array,
                             _mean: Array,
                             _cov: Array,
                             _meas: Array,
                             _mask: Array):
        cluster_count = jnp.sum(_mask)

        _x_deviation = _samples - _mean
        _h_samples = measurement_function(_samples)

        _h_sum = jnp.sum(jnp.where(_mask[:, None], _h_samples, 0.0), axis=0)
        _h_mean = _h_sum / jnp.maximum(cluster_count, 1)

        _h_deviation = _h_samples - _h_mean

        outer_yx = outer(_h_deviation, _x_deviation)
        masked_outer_yx = jnp.where(_mask[:, None, None], outer_yx, 0.0)
        P_yx = jnp.sum(masked_outer_yx, axis=0) / jnp.maximum(cluster_count, 1)

        outer_yy = outer(_h_deviation, _h_deviation)
        masked_outer_yy = jnp.where(_mask[:, None, None], outer_yy, 0.0)
        P_yy = jnp.sum(masked_outer_yy, axis=0) / jnp.maximum(cluster_count, 1) + measurement_covariance

        K = (jnp.linalg.solve(P_yy, P_yx)).T
        _mean = _mean + K @ (_meas - _h_mean)
        _cov = _cov - K @ P_yx

        return _mean, _cov

    def apply_kalman_update_to_cluster(j: int, samples: Array, assignments: Array,
                                       mean: Array, cov: Array, meas: Array):
        mask = (assignments == j)
        mean_updated, cov_updated = kalman_update_masked(samples, mean, cov, meas, mask)
        return mean_updated, cov_updated

    @filter_jit
    def pgm_update(samples: Array, meas: Array, prng_key: Array):
        # Step 1: K-means clustering
        means, covs, weights, assignments = k_means_with_assignments(
            samples,
            cluster_dim,
            rng=prng_key,
            init='k-means++',
            max_iterations=kmeans_max_iterations,
            tol=kmeans_tol,
        )

        # Step 2: Kalman update for all clusters
        cluster_indices = jnp.arange(cluster_dim)

        def update_cluster(j, mean, cov):
            return apply_kalman_update_to_cluster(j, samples, assignments, mean, cov, meas)

        means, covs = jax.vmap(update_cluster)(cluster_indices, means, covs)
        return means, covs, weights

    return pgm_update

