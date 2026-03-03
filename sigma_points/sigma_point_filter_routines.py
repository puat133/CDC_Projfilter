from equinox import filter_jit
from sigma_points.sigma_points import SigmaPoints
import jax.numpy as jnp
from typing import Callable
from functools import partial
from jaxtyping import Array

@partial(jnp.vectorize, signature="(n),(m)->(n,m)")
def outer(x: Array, y: Array) -> Array:
    """
    Vectorized numpy outer product.
    Parameters
    ----------
    x : np.ndarray
        First argument.
    y : np.ndarray
        Second argument.

    Returns
    -------
    out : np.ndarray
        The Outer product.
    """

    return jnp.outer(x, y)


@filter_jit
def empirical_covariance(xs: Array, ys: Array, weight: Array) -> Array:
    """
    Compute covariance

    Parameters
    ----------
    xs : Array
        First argument
    ys : Array
        Second argument
    weight : Array
        Weighting

    Returns
    -------
    out: Array
        The empirical covariance.
    """
    non_normalized = outer(xs, ys)
    return jnp.tensordot(weight, non_normalized, axes=([0], [0]))


@filter_jit
def update_step(
        sigma_point: SigmaPoints,
        measurement_fun: Callable[[Array], Array],
        mean_apriori: Array,
        cov_apriori: Array,
        meas_cov: Array,
        meas: Array,

) -> tuple:
    """
    Perform the update step of the Sigma Point Kalman Filter.

    Parameters
    ----------
    sigma_point : SigmaPoints
        Sigma point generator
    measurement_fun : Callable[[Array], Array]
        Measurement function
    mean_apriori : Array
        Prior mean
    cov_apriori : Array
        Prior covariance
    meas_cov : Array
        Measurement covariance
    meas : Array
        Measurement

    Returns
    -------
    mean_posterior : Array
        Posterior mean
    cov_posterior : Array
        Posterior covariance
    """
    # update
    state_ust = sigma_point.get_sigma_points(mean_apriori, cov_apriori)
    h_ust = measurement_fun(state_ust)

    h_ = sigma_point.weights_mean @ h_ust
    S_ = empirical_covariance(h_ust - h_, h_ust - h_, sigma_point.weights_cov) + meas_cov
    cov_h_f = empirical_covariance(h_ust - h_, state_ust - mean_apriori, sigma_point.weights_cov)
    gain_transp = jnp.linalg.solve(S_, cov_h_f)

    # The remaining is just the ordinary kalman filter
    cov_posterior = cov_apriori - gain_transp.T @ S_ @ gain_transp
    y_tilde = (meas - h_)
    mean_posterior = mean_apriori + y_tilde @ gain_transp

    return mean_posterior, cov_posterior
