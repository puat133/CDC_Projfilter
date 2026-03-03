from typing import Callable

import jax
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
import ott.tools.gaussian_mixture.gaussian_mixture as gm
from diffrax import AbstractSolver
from other_filter.fast_em import em
from equinox import filter_jit
from jax.lax import scan
from sigma_points.sigma_point_filter_routines import outer
from utils.diffrax import VectorizedControlTerm
from jaxtyping import PyTree, Array, Float


@filter_jit
def cd_pgm(process_drift: Callable[[Float, Array, PyTree], Array],
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
           em_max_iterations: int = 10,
           em_tol: Float = 1e-4,
           dt: Float = 1e-3):
    """
    CD-PGM Filter Implementation using EM fitting and JAX scan.

    Parameters
    ----------
    process_drift : Callable[[Float, ndarray, PyTree], ndarray]
        The drift function of the SDE. Takes time, state, *args as inputs.
    process_diffusion : Callable[[Float, ndarray, PyTree], ndarray]
        The diffusion function of the SDE. Takes time, state, *args as inputs.
    measurement_function : Callable[[ndarray], ndarray]
        Function mapping states to measurements.
    measurement_covariance : ndarray
        The measurement noise covariance matrix.
    time_sample : Float
        The time step between measurements.
    means_init : ndarray
        Initial means for each Gaussian component.
    covs_init : ndarray
        Initial covariances for each Gaussian component.
    n_samples : int
        Number of particles to use.
    weights_init : ndarray
        Initial weights for each Gaussian component.
    meas_record : ndarray
        Array of measurements.
    process_brownian_dim : int
        Dimension of the Brownian motion.
    prng_key : ndarray
        JAX PRNG key.
    sde_solver : AbstractSolver, optional
        The SDE solver to use, by default EulerHeun()
    em_max_iterations : int, optional
        Maximum number of EM iterations, by default 10
    em_tol : Float, optional
        Convergence tolerance for EM (stops when |ll_new - ll_old| < em_tol), by default 1e-4
    dt : Float, optional
        Time step for SDE solver, by default 1e-3

    Returns
    -------
    tuple
        means_hist : Array of component means over time
        covs_hist : Array of component covariances over time
        weights_hist : Array of component weights over time
        samples_sol_hist : diffeqsolve solution history (for stats)

    Notes
    -----
    Based on Algorithm I in "Particle Gaussian mixture filters-I"
    DOI 10.1016/j.automatica.2018.07.023
    Continuous-Discrete Particle Gauss Mixture Filter with EM fitting
    """

    cluster_dim = means_init.shape[0]
    state_dim = means_init.shape[1]

    def kalman_update_masked(_samples: Array,
                             _mean: Array,
                             _cov: Array,
                             _meas: Array,
                             _mask: Array):
        """
        Kalman update using masked samples for a specific cluster.

        Parameters
        ----------
        _samples : Array
            All samples, shape (n_samples, state_dim)
        _mean : Array
            Cluster mean from EM fit, shape (state_dim,)
        _cov : Array
            Cluster covariance from EM fit, shape (state_dim, state_dim)
        _meas : Array
            Measurement, shape (meas_dim,)
        _mask : Array
            Boolean mask for samples in this cluster, shape (n_samples,)

        Returns
        -------
        Tuple[Array, Array]
            Updated mean and covariance
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
        # P_yx = E[(h - h_mean)(x - mean)^T]
        outer_yx = outer(_h_deviation, _x_deviation)  # (n_samples, meas_dim, state_dim)
        masked_outer_yx = jnp.where(_mask[:, None, None], outer_yx, 0.0)
        P_yx = jnp.sum(masked_outer_yx, axis=0) / jnp.maximum(cluster_count, 1)

        # P_yy = E[(h - h_mean)(h - h_mean)^T]
        outer_yy = outer(_h_deviation, _h_deviation)  # (n_samples, meas_dim, meas_dim)
        masked_outer_yy = jnp.where(_mask[:, None, None], outer_yy, 0.0)
        P_yy = jnp.sum(masked_outer_yy, axis=0) / jnp.maximum(cluster_count, 1) + measurement_covariance

        K = (jnp.linalg.solve(P_yy, P_yx)).T
        _mean = _mean + K @ (_meas - _h_mean)
        _cov = _cov - K @ P_yx

        return _mean, _cov

    def process_single_cluster(j: int, samples: Array, assignments: Array, meas: Array,
                               gmm_loc: Array, gmm_cov: Array):
        """
        Process a single cluster: use EM-fitted params and apply Kalman update.

        Parameters
        ----------
        j : int
            Cluster index
        samples : Array
            All propagated samples, shape (n_samples, state_dim)
        assignments : Array
            Cluster assignments for each sample, shape (n_samples,)
        meas : Array
            Current measurement
        gmm_loc : Array
            EM-fitted means for all clusters, shape (cluster_dim, state_dim)
        gmm_cov : Array
            EM-fitted covariances for all clusters, shape (cluster_dim, state_dim, state_dim)

        Returns
        -------
        Tuple[Array, Array, Float]
            Updated mean, covariance, and weight for this cluster
        """
        mask = (assignments == j)
        cluster_count = jnp.sum(mask)
        weight = cluster_count / n_samples

        # Use EM-fitted mean and covariance
        mean = gmm_loc[j]
        cov = gmm_cov[j]

        # Kalman update
        mean, cov = kalman_update_masked(samples, mean, cov, meas, mask)

        # Handle NaN by keeping original values (using jnp.where for JIT compatibility)
        has_nan = jnp.any(jnp.isnan(mean)) | jnp.any(jnp.isnan(cov))
        mean = jnp.where(has_nan, gmm_loc[j], mean)
        cov = jnp.where(has_nan, gmm_cov[j], cov)

        return mean, cov, weight

    def scanned_fun(_carry, _input):
        _prng_key, _means, _covs, _weights, _t = _carry
        _meas = _input

        # Sample from GMM
        _prng_key, subkey = jrandom.split(_prng_key)
        _gmm = gm.GaussianMixture.from_mean_cov_component_weights(_means, _covs, _weights)
        samples = _gmm.sample(subkey, n_samples)

        # Propagate samples through SDE
        _prng_key, subkey = jrandom.split(_prng_key)
        brownian_motion = dfx.UnsafeBrownianPath(shape=(n_samples, process_brownian_dim,), key=subkey,
                                                 levy_area=dfx.BrownianIncrement)
        sde_terms = dfx.MultiTerm(dfx.ODETerm(process_drift),
                                  VectorizedControlTerm(process_diffusion,
                                                        brownian_motion))

        samples_sol = dfx.diffeqsolve(sde_terms,
                                      sde_solver,
                                      _t,  # starting time
                                      _t + time_sample,  # end time
                                      dt0=dt,  # sde solver delta t
                                      y0=samples,
                                      saveat=dfx.SaveAt(t1=True),  # only saves at the end
                                      adjoint=dfx.DirectAdjoint())

        samples = samples_sol.ys[-1]

        # Fit GMM using fast EM (replaces OTT fit_model_em)
        fitted_loc, fitted_cov, fitted_weights = em(samples, em_max_iterations, _means, _covs, _weights, em_tol)

        # Get cluster assignments from posterior
        # Recreate GMM with fitted parameters to compute posteriors
        fitted_gmm = gm.GaussianMixture.from_mean_cov_component_weights(
            fitted_loc, fitted_cov, fitted_weights
        )
        logit = fitted_gmm.get_log_component_posterior(samples)
        _prng_key, subkey = jrandom.split(_prng_key)
        assignments = jrandom.categorical(subkey, logit)

        # Process all clusters using vmap
        cluster_indices = jnp.arange(cluster_dim)

        def process_cluster_wrapper(j):
            return process_single_cluster(j, samples, assignments, _meas,
                                          fitted_loc, fitted_cov)

        _means, _covs, _weights = jax.vmap(process_cluster_wrapper)(cluster_indices)

        _carry = (_prng_key, _means, _covs, _weights, _t + time_sample)
        return _carry, (_means, _covs, _weights, samples_sol)

    init_carry = (prng_key, means_init, covs_init, weights_init, 0.0)
    _, results = scan(scanned_fun, init_carry, xs=meas_record)

    return results


def create_pgm_em_drift_for_cost_analysis(
    process_drift: Callable[[Float, Array, PyTree], Array],
) -> Callable:
    """
    Create a standalone JIT-compiled PGM-EM drift function for cost analysis.

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


def create_pgm_em_update_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    cluster_dim: int,
    n_samples: int,
    em_max_iterations: int = 10,
    em_tol: float = 1e-4,
) -> Callable:
    """
    Create a standalone JIT-compiled PGM-EM update function for cost analysis.

    This measures the full cost of the PGM-EM update step including:
    1. EM fitting to get GMM parameters
    2. Computing cluster assignments from posterior
    3. Kalman update for all clusters

    Parameters
    ----------
    measurement_function : Callable[[Array], Array]
        Function that maps state to measurement space.
    measurement_covariance : Array
        Covariance matrix of measurement noise.
    cluster_dim : int
        Number of Gaussian mixture components.
    n_samples : int
        Number of particles.
    em_max_iterations : int
        Maximum number of EM iterations. Default is 10.
    em_tol : float
        Convergence tolerance for EM. Default is 1e-4.

    Returns
    -------
    Callable
        JIT-compiled update function with signature:
        (samples, means_init, covs_init, weights_init, meas, prng_key) -> (means, covs, weights)
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

    def process_single_cluster(j: int, samples: Array, assignments: Array, meas: Array,
                               gmm_loc: Array, gmm_cov: Array):
        mask = (assignments == j)
        cluster_count = jnp.sum(mask)
        weight = cluster_count / n_samples

        mean = gmm_loc[j]
        cov = gmm_cov[j]

        mean, cov = kalman_update_masked(samples, mean, cov, meas, mask)

        has_nan = jnp.any(jnp.isnan(mean)) | jnp.any(jnp.isnan(cov))
        mean = jnp.where(has_nan, gmm_loc[j], mean)
        cov = jnp.where(has_nan, gmm_cov[j], cov)

        return mean, cov, weight

    @filter_jit
    def pgm_em_update(samples: Array, means_init: Array, covs_init: Array,
                      weights_init: Array, meas: Array, prng_key: Array):
        # Step 1: EM fitting
        fitted_loc, fitted_cov, fitted_weights = em(samples, em_max_iterations, means_init, covs_init, weights_init, em_tol)

        # Step 2: Get cluster assignments from posterior
        fitted_gmm = gm.GaussianMixture.from_mean_cov_component_weights(
            fitted_loc, fitted_cov, fitted_weights
        )
        logit = fitted_gmm.get_log_component_posterior(samples)
        assignments = jrandom.categorical(prng_key, logit)

        # Step 3: Kalman update for all clusters
        cluster_indices = jnp.arange(cluster_dim)

        def process_cluster_wrapper(j):
            return process_single_cluster(j, samples, assignments, meas, fitted_loc, fitted_cov)

        means, covs, weights = jax.vmap(process_cluster_wrapper)(cluster_indices)
        return means, covs, weights

    return pgm_em_update
