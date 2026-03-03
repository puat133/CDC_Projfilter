from typing import Callable

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import AbstractSolver
from equinox import filter_jit
from jax.lax import scan
import jax
from sigma_points.sigma_point_filter_routines import outer
from utils.diffrax import VectorizedControlTerm
from jaxtyping import PyTree, Array, Float

@filter_jit
def cd_enkf(process_drift: Callable[[Float, Array, PyTree], Array],
            process_diffusion: Callable[[Float, Array, PyTree], Array],
            measurement_function: Callable[[Array], Array],
            measurement_covariance: Array,
            time_sample: Float, samples_init: Array,
            meas_record: Array,
            n_devices: int,
            n_particle_per_device: int,
            process_brownian_dim: int,
            prng_key: Array,
            sde_solver: AbstractSolver = dfx.EulerHeun(),
            dt: Float = 1e-3):
    """
    Implements a Continuous-Discrete Ensemble Kalman Filter for filtering in a nonlinear state-space model with continuous
    state evolution and discrete measurements.

    The algorithm propagates an ensemble of particles through the continuous time state model using SDE integration,
    then performs a Kalman update step when measurements arrive.

    The state process should follow an Ito SDE of the form:
        dx = f(t,x)dt + L(t,x)dW

    With discrete measurements:
        y_k = h(x_k) + v_k
        v_k ~ N(0,R)

    Continuous-Discrete Ensemble Kalman Filter, based on:
    Section III of Ensemble Kalman Filter for Continuous-Discrete State-Space Models
    DOI 10.1109/CDC45484.2021.9682835


    Parameters
    ----------
    process_drift: callable
        Function representing the drift term of the SDE. Takes current time, state, and parameters as input.
    process_diffusion: callable
        Function representing the diffusion term of the SDE. Takes current time, state, and parameters as input.
    measurement_function: callable
        Function mapping states to measurements. Takes state as input.
    measurement_covariance: array_like
        Measurement noise covariance matrix
    time_sample: Float
        Time interval between measurements
    samples_init: array_like
        Initial ensemble of samples/particles
    meas_record: array_like
        Array of measurements over time
    n_devices: int
        Number of parallel devices for computation
    n_particle_per_device: int
        Number of particles per device
    process_brownian_dim: int
        Dimension of the Brownian motion
    prng_key: array_like
        Random number generator key
    sde_solver: AbstractSolver, optional
        SDE solver to use, defaults to EulerHeun
    dt: Float, optional
        Time step for SDE solver, defaults to 1e-3

    Returns
    -------
    array_like
        Array of filtered state estimates over time

    """

    def scanned_fun(_carry, _input):
        _meas = _input
        _prng_key, _samples, _t = _carry

        # solve the SDE
        _prng_key, _subkey = jrandom.split(_prng_key)
        _brownian_motion = dfx.UnsafeBrownianPath(shape=(n_devices, n_particle_per_device,
                                                         process_brownian_dim,), key=_subkey,
                                                  levy_area=dfx.BrownianIncrement)
        # propagate according to the sde
        sde_terms = dfx.MultiTerm(dfx.ODETerm(process_drift),
                                  VectorizedControlTerm(process_diffusion,
                                                        _brownian_motion))

        samples_sol = dfx.diffeqsolve(sde_terms,
                                      sde_solver,
                                      _t,  # starting time
                                      _t + time_sample,  # end time
                                      dt0=dt,  # sde solver delta t
                                      y0=_samples,
                                      saveat=dfx.SaveAt(t1=True),  # only saves at the end
                                      adjoint=dfx.DirectAdjoint())

        _samples = samples_sol.ys[-1]

        # Kalman Filter Update

        sampled_meas = measurement_function(_samples)
        meas_mean = sampled_meas.mean(axis=(0, 1))

        states_diffs = _samples - _samples.mean(axis=(0, 1))
        meas_diffs = sampled_meas - meas_mean

        # calculate covariances
        cov_yy = (jnp.sum(outer(meas_diffs, meas_diffs), axis=(0, 1)) / (n_devices * n_particle_per_device - 1)
                  + measurement_covariance)
        cov_yx = jnp.sum(outer(meas_diffs, states_diffs), axis=(0, 1)) / (n_devices * n_particle_per_device - 1)
        kalman_gain = jnp.linalg.solve(cov_yy, cov_yx).T
        # jax.debug.print('The shape of kalman gain is {}, _meas is {}, and sampled meas is {}'.format(
        #     kalman_gain.shape,_meas.shape,sampled_meas.shape)
        #           )

        # update the samples according to the Kalman update
        _prng_key, _subkey = jrandom.split(_prng_key)
        # _meas_noise_samples is needed according to eq. 4.37 from Data Assimilation  The Ensemble Kalman Filter 2nd Ed.
        # although the difference is not that much
        _meas_noise_samples = jrandom.multivariate_normal(_subkey, jnp.zeros(_meas.shape[0]), measurement_covariance,
                                                          (n_devices, n_particle_per_device))
        _samples = _samples + jnp.einsum('ij,...j', kalman_gain, (_meas + _meas_noise_samples - sampled_meas))

        _carry = _prng_key, _samples, _t + time_sample

        return _carry, (_samples, samples_sol)

    _, results = scan(scanned_fun, (prng_key, samples_init, 0.), xs=meas_record)
    return results


def create_enkf_drift_for_cost_analysis(
    process_drift: Callable[[Float, Array, PyTree], Array],
) -> Callable:
    """
    Create a standalone JIT-compiled EnKF drift function for cost analysis.

    For EnKF, we measure the drift term separately since the SDE uses MultiTerm.
    The diffusion term cost is typically dominated by the drift for large ensembles.

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


def create_enkf_update_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    n_devices: int,
    n_particle_per_device: int,
) -> Callable:
    """
    Create a standalone JIT-compiled EnKF Bayesian update function for cost analysis.

    Parameters
    ----------
    measurement_function : Callable[[Array], Array]
        The measurement function h(x).
    measurement_covariance : Array
        The measurement noise covariance matrix R.
    n_devices : int
        Number of parallel devices.
    n_particle_per_device : int
        Number of particles per device.

    Returns
    -------
    Callable
        JIT-compiled update function with signature (samples, meas, meas_noise_samples) -> samples_post
    """
    @filter_jit
    def enkf_update(samples: Array, meas: Array, meas_noise_samples: Array):
        sampled_meas = measurement_function(samples)
        meas_mean = sampled_meas.mean(axis=(0, 1))

        states_diffs = samples - samples.mean(axis=(0, 1))
        meas_diffs = sampled_meas - meas_mean

        # calculate covariances
        cov_yy = (jnp.sum(outer(meas_diffs, meas_diffs), axis=(0, 1)) / (n_devices * n_particle_per_device - 1)
                  + measurement_covariance)
        cov_yx = jnp.sum(outer(meas_diffs, states_diffs), axis=(0, 1)) / (n_devices * n_particle_per_device - 1)
        kalman_gain = jnp.linalg.solve(cov_yy, cov_yx).T

        # update the samples according to the Kalman update
        samples_post = samples + jnp.einsum('ij,...j', kalman_gain, (meas + meas_noise_samples - sampled_meas))

        return samples_post

    return enkf_update
