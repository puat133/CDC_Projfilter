from functools import partial
from typing import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import AbstractSolver
from equinox import Module, filter_jit
from jax.lax import scan
from jaxtyping import PyTree, Array
from other_filter.resampling import log_gaussian_density, normalize_log_weights


@filter_jit
@partial(jnp.vectorize, signature='(n,n),(n),(n,m),(n,n)->(n),(n,n)')
def ode_mean_and_cov(cov: Array, f_eval: Array, g_eval: Array, fjac_eval: Array):
    """
    Applies the ODE mean and covariance update.

    Parameters
    ----------
    cov : Array
        The covariance matrix at current time step
    f_eval : Array
        Evaluation of drift function
    g_eval : Array
        Evaluation of diffusion function
    fjac_eval : Array
        Evaluation of drift function jacobian

    Returns
    -------
    Tuple
        The ODE mean and covariance update values
    """
    fjac_eval_times_cov = fjac_eval @ cov
    return f_eval, fjac_eval_times_cov + fjac_eval_times_cov.T + g_eval @ g_eval.T


@filter_jit
@partial(jnp.vectorize, signature='(n),(n,n),(m),(m,n)->(n),(n,n)', excluded=(4, 5))
def bayesian_update(mean: Array, cov: Array, heval: Array, hjac_eval: Array,
                    meas: Array, meas_cov: Array):
    """
    Updates the filter mean and covariance by applying a Bayesian update.

    Parameters
    ----------
    mean : Array
        The current state mean estimate
    cov : Array
        The current state covariance estimate
    heval : Array
        Evaluation of measurement function
    hjac_eval : Array
        Evaluation of measurement function jacobian
    meas : Array
        The measurement
    meas_cov : Array
        The measurement noise covariance

    Returns
    -------
    Tuple[Array, Array]
        The updated mean and covariance
    """
    _K_T = jnp.linalg.solve(hjac_eval @ cov @ hjac_eval.T + meas_cov, hjac_eval @ cov)

    mean = mean + _K_T.T @ (meas - heval)
    cov = (jnp.eye(mean.shape[0]) - _K_T.T @ hjac_eval) @ cov
    cov = 0.5 * (cov + cov.T)
    return mean, cov


class EKFFlow(Module):
    """
    An Extended Kalman Filter implementation which numerically calculates the continuous-discrete filter equations.

    Parameters
    ----------
    state_drift : Callable[[float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE
    state_drift_jac : Callable[[float, Array, PyTree], Array]
        The Jacobian of the drift function with respect to state
    state_diffusion : Callable[[float, Array, PyTree], Array]
        The diffusion function g(t, x, args) of the SDE

    Notes
    -----
    The EKF is assumed to work with the following SDE:

    dx_t = f(x) dt + g(x) dW,
    y_k = h(x_k) + v_k
    where
    - W is a vector of Wiener processes with E[dWdW^top] = dt
    - x_k = x_(k Delta t), with Delta t > 0
    - v_k is a measurement noise vector with covariance R


    """

    state_drift: Callable[[float, Array, PyTree], Array]
    state_drift_jac: Callable[[float, Array, PyTree], Array]
    state_diffusion: Callable[[float, Array, PyTree], Array]

    def __call__(self,
                 t,
                 states,
                 args
                 ) -> tuple:
        _mu, _P = states
        # the actual state_drift and state_diffusion do not depend on t and args
        args = (0, _mu, None)
        _f = self.state_drift(*args)
        _F = self.state_drift_jac(*args)
        _g = self.state_diffusion(*args)
        d_mean, d_cov = ode_mean_and_cov(_P, _f, _g, _F)
        return d_mean, d_cov


@filter_jit
def cd_ekf(process_drift: Callable[[float, Array, PyTree], Array],
           process_diffusion: Callable[[float, Array, PyTree], Array],
           measurement_function: Callable[[Array], Array],
           measurement_covariance: Array,
           time_sample: float,
           mean_init: Array,
           cov_init: Array,
           meas_record: Array,
           ode_solver: AbstractSolver = dfx.Tsit5(),
           constant_step_size: bool = False,
           rtol: float = 1e-3,
           atol: float = 1e-8,
           dt: float = 1e-3,
           ):
    """
    A continuous-discrete extended Kalman filter implementation.

    Parameters
    ----------
    process_drift : Callable[[float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE
    process_diffusion : Callable[[float, Array, PyTree], Array]
        The diffusion function g(t, x, args) of the SDE
    measurement_function : Callable[[Array], Array]
        The measurement function h(x)
    measurement_covariance : Array
        The measurement noise covariance matrix R
    time_sample : float
        The time interval between measurements
    mean_init : Array
        Initial state mean
    cov_init : Array
        Initial state covariance
    meas_record : Array
        Array of measurements
    ode_solver : AbstractSolver, optional
        The ODE solver to use, by default Tsit5
    constant_step_size : bool, optional
        Whether to use constant step size in ODE solver, by default False
    rtol : float, optional
        Relative tolerance for adaptive step size, by default 1e-3
    atol : float, optional
        Absolute tolerance for adaptive step size, by default 1e-8
    dt : float, optional
        Step size (if constant) or initial step size, by default 1e-3

    Returns
    -------
    tuple
        Tuple containing the filtered state means and covariances
    """

    state_drift_jac = jax.jacobian(process_drift, argnums=1)
    meas_fun_jac = jax.jacobian(measurement_function)
    ode_term = dfx.ODETerm(EKFFlow(process_drift, state_drift_jac, process_diffusion))
    saveat = dfx.SaveAt(t1=True)  # Only save the last one
    if constant_step_size:
        stepsize_controller = dfx.ConstantStepSize()
    else:
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)

    t_0 = 0.  # start at 0
    _I = jnp.eye(mean_init.shape[0])

    def scanned_fun(_carry, _input):
        _t_0, _mu, _P = _carry
        meas = _input

        # The prediction steps is done by solving the differential equation for mean and covariance
        initial_states = (_mu, _P)
        a_sol = dfx.diffeqsolve(
            ode_term,
            ode_solver,
            _t_0,
            _t_0 + time_sample,
            dt0=dt,
            y0=initial_states,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        _mu, _P = a_sol.ys
        _mu = _mu.squeeze()
        _P = _P.squeeze()

        # Bayesian update
        _h = measurement_function(_mu)
        _H = meas_fun_jac(_mu)
        _mu, _P = bayesian_update(_mu, _P, _h, _H, meas, measurement_covariance)

        _carry = _t_0 + time_sample, _mu, _P
        return _carry, (_mu, _P)

    _, results = scan(scanned_fun, (t_0, mean_init, cov_init), xs=meas_record)
    return results


@filter_jit
def cd_gsf(process_drift: Callable[[float, Array, PyTree], Array],
           process_diffusion: Callable[[float, Array, PyTree], Array],
           measurement_function: Callable[[Array], Array],
           measurement_covariance: Array,
           time_sample: float,
           means_init: Array,
           covs_init: Array,
           log_weights_init: Array,
           meas_record: Array,
           ode_solver: AbstractSolver = dfx.Tsit5(),
           constant_step_size: bool = False,
           rtol: float = 1e-3,
           atol: float = 1e-8,
           dt: float = 1e-3,
           ):
    """
    A continuous-discrete Gaussian Sum Filter implementation. The GSF is a mixture of Extended Kalman Filters which
    approximates arbitrary probability distributions as a weighted sum of Gaussian distributions.

    Like the EKF, the GSF works with the following SDE:

    dx_t = f(x) dt + g(x) dW,
    y_k = h(x_k) + v_k
    where
    - W is a vector of Wiener processes with E[dWdW^top] = dt
    - x_k = x_(k Delta t), with Delta t > 0
    - v_k is a measurement noise vector with covariance R

    Each component in the mixture evolves according to EKF equations, with weights updated based on measurement likelihoods.

    Parameters
    ----------
    process_drift : Callable[[float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE
    process_diffusion : Callable[[float, Array, PyTree], Array]
        The diffusion function g(t, x, args) of the SDE
    measurement_function : Callable[[Array], Array]
        The measurement function h(x)
    measurement_covariance : Array
        The measurement noise covariance matrix R
    time_sample : float
        The time interval between measurements
    means_init : Array
        Initial state means for each mixture component
    covs_init : Array
        Initial state covariances for each mixture component
    log_weights_init : Array
        Initial log weights for each mixture component
    meas_record : Array
        Array of measurements
    ode_solver : AbstractSolver, optional
        The ODE solver to use, by default Tsit5
    constant_step_size : bool, optional
        Whether to use constant step size in ODE solver, by default False
    rtol : float, optional
        Relative tolerance for adaptive step size, by default 1e-3
    atol : float, optional
        Absolute tolerance for adaptive step size, by default 1e-8
    dt : float, optional
        Step size (if constant) or initial step size, by default 1e-3

    Returns
    -------
    tuple
        Tuple containing the filtered state means, covariances and log weights for each mixture component
    """
    state_drift_jac = jax.jacobian(process_drift, argnums=1)
    state_drift_jac = jnp.vectorize(state_drift_jac, signature="(n)->(n,n)", excluded=(0, 2))
    meas_fun_jac = jax.jacobian(measurement_function)
    meas_fun_jac = jnp.vectorize(meas_fun_jac, signature="(n)->(m,n)")
    ode_term = dfx.ODETerm(EKFFlow(process_drift, state_drift_jac, process_diffusion))
    saveat = dfx.SaveAt(t1=True)  # Only save the last one
    if constant_step_size:
        stepsize_controller = dfx.ConstantStepSize()
    else:
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)

    t_0 = 0.  # start at 0
    _I = jnp.eye(means_init.shape[0])

    def scanned_fun(_carry, _input):
        _t_0, _mus, _Ps, _log_weights = _carry
        meas = _input

        # The prediction steps is done by solving the differential equation for mean and covariance
        initial_states = (_mus, _Ps)
        a_sol = dfx.diffeqsolve(
            ode_term,
            ode_solver,
            _t_0,
            _t_0 + time_sample,
            dt0=dt,
            y0=initial_states,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
        )
        _mus, _Ps = a_sol.ys
        _mus = _mus.squeeze()
        _Ps = _Ps.squeeze()

        # Bayesian update
        _hs = measurement_function(_mus)
        _Hs = meas_fun_jac(_mus)
        _mus, _Ps = bayesian_update(_mus, _Ps, _hs, _Hs, meas, measurement_covariance)

        # compute log weights update
        log_weight_evidence = log_gaussian_density(meas - _hs,
                                                   _Hs @ _Ps @ _Hs.transpose(0, 2, 1) + measurement_covariance)
        _log_weights = normalize_log_weights(log_weight_evidence + _log_weights)

        _carry = _t_0 + time_sample, _mus, _Ps, _log_weights

        return _carry, (_mus, _Ps, _log_weights, a_sol)

    _, results = scan(scanned_fun, (t_0, means_init, covs_init, log_weights_init), xs=meas_record)
    return results


def create_ekf_ode_for_cost_analysis(
    process_drift: Callable[[float, Array, PyTree], Array],
    process_diffusion: Callable[[float, Array, PyTree], Array],
) -> Callable:
    """
    Create a standalone JIT-compiled EKF ODE derivative function for cost analysis.

    This function creates a standalone version of the EKFFlow.__call__
    that can be compiled and analyzed independently to measure FLOPS per ODE step.

    Parameters
    ----------
    process_drift : Callable[[float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE.
    process_diffusion : Callable[[float, Array, PyTree], Array]
        The diffusion function g(t, x, args) of the SDE.

    Returns
    -------
    Callable
        JIT-compiled ODE derivative function with signature (t, states, args) -> derivatives
    """
    state_drift_jac = jax.jacobian(process_drift, argnums=1)
    state_drift_jac = jnp.vectorize(state_drift_jac, signature="(n)->(n,n)", excluded=(0, 2))

    @filter_jit
    def ode_derivative(t, states, args=None):
        _mus, _Ps = states
        # the actual state_drift and state_diffusion do not depend on t and args
        _args = (0, _mus, None)
        _f = process_drift(*_args)
        _F = state_drift_jac(*_args)
        _g = process_diffusion(*_args)
        d_mean, d_cov = ode_mean_and_cov(_Ps, _f, _g, _F)
        return d_mean, d_cov

    return ode_derivative


def create_gsf_update_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
) -> Callable:
    """
    Create a standalone JIT-compiled GSF Bayesian update function for cost analysis.

    This function creates a standalone version of the GSF Bayesian update step
    that can be compiled and analyzed independently to measure FLOPS per measurement update.

    The update consists of:
    1. Computing measurement predictions h(mu) and Jacobians H
    2. Applying EKF update to each component
    3. Updating log weights based on measurement likelihoods

    Parameters
    ----------
    measurement_function : Callable[[Array], Array]
        The measurement function h(x).
    measurement_covariance : Array
        The measurement noise covariance matrix R.

    Returns
    -------
    Callable
        JIT-compiled update function with signature (mus, Ps, log_weights, meas) -> (mus, Ps, log_weights)
    """
    meas_fun_jac = jax.jacobian(measurement_function)
    meas_fun_jac = jnp.vectorize(meas_fun_jac, signature="(n)->(m,n)")

    @filter_jit
    def gsf_update(mus: Array, Ps: Array, log_weights: Array, meas: Array):
        # Bayesian update
        _hs = measurement_function(mus)
        _Hs = meas_fun_jac(mus)
        mus_post, Ps_post = bayesian_update(mus, Ps, _hs, _Hs, meas, measurement_covariance)

        # compute log weights update
        log_weight_evidence = log_gaussian_density(meas - _hs,
                                                   _Hs @ Ps @ _Hs.transpose(0, 2, 1) + measurement_covariance)
        log_weights_post = normalize_log_weights(log_weight_evidence + log_weights)

        return mus_post, Ps_post, log_weights_post

    return gsf_update
