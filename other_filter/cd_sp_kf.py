from functools import partial
from typing import Callable

import diffrax as dfx
import jax.numpy as jnp
from diffrax import AbstractSolver
from equinox import Module, filter_jit
from jax.lax import scan

from other_filter.resampling import normalize_log_weights
from sigma_points.sigma_point_filter_routines import outer, empirical_covariance
from sigma_points.sigma_points import SigmaPoints
from sigma_points.unscented import UnscentedSigmaPoints
from jaxtyping import PyTree, Array, Float


@filter_jit
@partial(jnp.vectorize, signature='(n),(n,n)->(n),(n,n)', excluded=(2, 3, 4))
def ode_mean_and_cov(mean: Array,
                     cov: Array,
                     process_diffusion: Callable[[Float, Array, PyTree], Array],
                     process_drift: Callable[[Float, Array, PyTree], Array],
                     sigma_points: SigmaPoints, ):
    """
    Given state-space mean and covariance, calculates ODEs for mean and covariance of OU-process type model.

    Parameters
    ----------
    mean : Array
        State-space mean vector
    cov : Array
        State-space covariance matrix
    process_diffusion : Callable
        Function defining diffusion term g(x)
    process_drift : Callable
        Function defining drift term f(x)
    sigma_points : SigmaPoints
        Sigma points object for unscented transform

    Returns
    -------
    f_mean : Array
        Drift contribution to ODE for mean
    P_mean : Array
        Drift and diffusion contribution to ODE for covariance
    """
    x_sp = sigma_points.get_sigma_points(mean, cov)
    f_sp = process_drift(0, x_sp, None)
    g_sp = process_diffusion(0, x_sp, None)

    f_mean = sigma_points.weights_mean @ f_sp

    temp = outer(x_sp - mean, f_sp) + outer(f_sp, x_sp - mean) + jnp.einsum('...ij,...kj', g_sp, g_sp)
    P_mean = jnp.einsum('i,i...', sigma_points.weights_cov, temp)
    return f_mean, P_mean


@filter_jit
@partial(jnp.vectorize, signature='(n),(n,n)->(n),(n,n),()', excluded=(2, 3, 4, 5))
def bayesian_update(mean: Array,
                    cov: Array,
                    meas: Array,
                    meas_cov: Array,
                    meas_function: Callable[[Array], Array],
                    sigma_point: SigmaPoints):
    """
    Performs Bayesian update for sigma points using the Kalman filter formulas.

    Parameters
    ----------
    mean : numpy.ndarray (n,)
        State-space mean vector
    cov : numpy.ndarray (n,n)
        State-space covariance matrix
    meas : numpy.ndarray (m,)
        Measurement vector
    meas_cov : numpy.ndarray (m,m)
        Measurement noise covariance matrix
    meas_function : callable
        Function that maps state to measurement
    sigma_point : SigmaPoints
        Sigma points object for unscented transform

    Returns
    -------
    mean_posterior : numpy.ndarray (n,)
        Updated state mean after measurement
    cov_posterior : numpy.ndarray (n,n)
        Updated state covariance after measurement
    un_normalized_log_likelihood : Float
        Unnormalized log likelihood of update
    """
    state_sp = sigma_point.get_sigma_points(mean, cov)
    h_sp = meas_function(state_sp)

    h_ = sigma_point.weights_mean @ h_sp
    S_ = empirical_covariance(h_sp - h_, h_sp - h_, sigma_point.weights_cov) + meas_cov
    cov_h_f = empirical_covariance(h_sp - h_, state_sp - mean, sigma_point.weights_cov)
    gain_transp = jnp.linalg.solve(S_, cov_h_f)

    # The remaining is just the ordinary kalman filter
    cov_posterior = cov - gain_transp.T @ S_ @ gain_transp
    y_tilde = (meas - h_)
    mean_posterior = mean + y_tilde @ gain_transp
    un_normalized_log_likelihood = y_tilde @ jnp.linalg.solve(S_, y_tilde)

    return mean_posterior, cov_posterior, un_normalized_log_likelihood


class SigmaPointKalmanFlow(Module):
    """
    Continuous-discrete time sigma point Kalman flow class for SDEs.

    A continuous-time SDE filtering class that uses sigma points to approximate the
    nonlinear transformations, similar to the unscented transform. This produces
    continuous-discrete time version of UKF-style filters.

    This ODE-based filter propagates sigma points and their dynamics according to the
    SDE model specification. The filter can handle non-additive noise.

    Parameters
    ----------
    state_drift : Callable[[Float, Array, PyTree], Array]
        Function defining drift term f(x)
    state_diffusion : Callable[[Float, Array, PyTree], Array]
        Function defining diffusion term g(x)
    sigma_points : SigmaPoints
        Sigma points object for unscented transform

    Notes
    -----
    The Sigma Point Kalman Flow is assumed to work with the following SDE:

    dx_t = f(x) dt + g(x) dW,
    y_k = h(x_k) + v_k
    where
    - W is a vector of Wiener processes with E[dWdW^top] = dt
    - x_k = x_(k Delta t), with Delta t > 0
    - v_k is a measurement noise vector with covariance R
    """

    state_drift: Callable[[Float, Array, PyTree], Array]
    state_diffusion: Callable[[Float, Array, PyTree], Array]
    sigma_points: SigmaPoints

    def __call__(self,
                 t,
                 states,
                 args
                 ) -> tuple:
        _mu, _P = states

        # the actual state_drift and state_diffusion do not depend on t and args
        d_mean, d_cov = ode_mean_and_cov(_mu, _P, self.state_diffusion, self.state_drift, self.sigma_points)
        return d_mean, d_cov


@filter_jit
def cd_ukf(process_drift: Callable[[Float, Array, PyTree], Array],
           process_diffusion: Callable[[Float, Array, PyTree], Array],
           measurement_function: Callable[[Array], Array],
           measurement_covariance: Array,
           time_sample: Float,
           mean_init: Array,
           cov_init: Array,
           meas_record: Array,
           ode_solver: AbstractSolver = dfx.Tsit5(),
           constant_step_size: bool = False,
           ukf_kappa: Float = 1.,
           ukf_alpha: Float = 1.,
           ukf_beta: Float = 0.,
           rtol: Float = 1e-3,
           atol: Float = 1e-8,
           dt: Float = 1e-3,
           ):
    """
    Continuous-discrete time unscented Kalman filter for SDEs.

    A continuous-time SDE filtering class that uses the unscented transform to approximate
    nonlinear transformations. This produces a continuous-discrete time version of the UKF.

    This ODE-based filter propagates mean and covariance according to ODEs derived from
    state space models. The filter can handle non-additive noise.

    Parameters
    ----------
    process_drift : Callable[[Float, Array, PyTree], Array]
        Function defining drift term f(x)
    process_diffusion : Callable[[Float, Array, PyTree], Array]
        Function defining diffusion term g(x)
    measurement_function : Callable[[Array], Array]
        Function that maps state to measurement
    measurement_covariance : Array
        Measurement noise covariance matrix R
    time_sample : Float
        Time between measurements
    mean_init : Array
        Initial state mean vector
    cov_init : Array
        Initial state covariance matrix
    meas_record : Array
        Array of measurements
    ode_solver : AbstractSolver, optional
        ODE solver to use for integrating mean and covariance ODEs
    constant_step_size : bool, optional
        Whether to use constant step size integration
    ukf_kappa : Float, optional
        UKF kappa parameter
    ukf_alpha : Float, optional
        UKF alpha parameter
    ukf_beta : Float, optional
        UKF beta parameter
    rtol : Float, optional
        Relative tolerance for ODE solver
    atol : Float, optional
        Absolute tolerance for ODE solver
    dt : Float, optional
        Initial time step for ODE solver

    Returns
    -------
    results : tuple
        Tuple containing filtered mean vectors and covariance matrices
    """

    sigma_points = UnscentedSigmaPoints(mean_init.shape[0], kappa=ukf_kappa, alpha=ukf_alpha, beta=ukf_beta)
    ode_term = dfx.ODETerm(SigmaPointKalmanFlow(process_drift, process_diffusion, sigma_points))
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
        _mu, _P, _ = bayesian_update(_mu, _P, meas, measurement_covariance, measurement_function, sigma_points)

        _carry = _t_0 + time_sample, _mu, _P
        return _carry, (_mu, _P)

    _, results = scan(scanned_fun, (t_0, mean_init, cov_init), xs=meas_record)
    return results


@filter_jit
def cd_sp_gsf(process_drift: Callable[[Float, Array, PyTree], Array],
              process_diffusion: Callable[[Float, Array, PyTree], Array],
              meas_fun: Callable[[Array], Array],
              meas_cov: Array,
              time_sample: Float,
              means_init: Array,
              covs_init: Array,
              log_weights_init: Array,
              meas_record: Array,
              sigma_points: SigmaPoints,
              ode_solver: AbstractSolver = dfx.Tsit5(),
              constant_step_size: bool = False,
              rtol: Float = 1e-3,
              atol: Float = 1e-8,
              dt: Float = 1e-3,
              ):
    """
    Continuous-discrete time sigma point Gaussian sum filter for SDEs.

    A continuous-time SDE filtering class that uses sigma points and a Gaussian sum approximation.
    Each component of the Gaussian mixture is propagated using sigma point ODEs. Updates are done
    using sigma point-based Kalman updates.

    This ODE-based filter propagates means and covariances according to ODEs derived from
    state space models. The filter can handle non-additive noise.

    Parameters
    ----------
    process_drift : Callable[[Float, Array, PyTree], Array]
        Function defining drift term f(x)
    process_diffusion : Callable[[Float, Array, PyTree], Array]
        Function defining diffusion term g(x)
    meas_fun : Callable[[Array], Array]
        Function that maps state to measurement
    meas_cov : Array
        Measurement noise covariance matrix R
    time_sample : Float
        Time between measurements
    means_init : Array
        Initial state mean vectors for mixture components
    covs_init : Array
        Initial state covariance matrices for mixture components
    log_weights_init : Array
        Initial log weights for mixture components
    meas_record : Array
        Array of measurements
    sigma_points : SigmaPoints
        Sigma points object for unscented transform
    ode_solver : AbstractSolver, optional
        ODE solver to use for integrating mean and covariance ODEs
    constant_step_size : bool, optional
        Whether to use constant step size integration
    rtol : Float, optional
        Relative tolerance for ODE solver
    atol : Float, optional
        Absolute tolerance for ODE solver
    dt : Float, optional
        Initial time step for ODE solver

    Returns
    -------
    results : tuple
        Tuple containing filtered mean vectors, covariance matrices and log weights
    """

    ode_term = dfx.ODETerm(SigmaPointKalmanFlow(process_drift, process_diffusion, sigma_points))
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

        _mus, _Ps, _log_weight_evidence = bayesian_update(_mus, _Ps, meas, meas_cov,
                                                          meas_fun, sigma_points)

        _log_weights = normalize_log_weights(_log_weight_evidence + _log_weights)

        _carry = _t_0 + time_sample, _mus, _Ps, _log_weights

        return _carry, (_mus, _Ps, _log_weights, a_sol)

    _, results = scan(scanned_fun, (t_0, means_init, covs_init, log_weights_init), xs=meas_record)
    return results


def create_spkf_ode_for_cost_analysis(
    process_drift: Callable[[Float, Array, PyTree], Array],
    process_diffusion: Callable[[Float, Array, PyTree], Array],
    sigma_points: SigmaPoints,
) -> Callable:
    """
    Create a standalone JIT-compiled SP-KF ODE derivative function for cost analysis.

    Parameters
    ----------
    process_drift : Callable[[Float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE.
    process_diffusion : Callable[[Float, Array, PyTree], Array]
        The diffusion function g(t, x, args) of the SDE.
    sigma_points : SigmaPoints
        Sigma points object for unscented transform.

    Returns
    -------
    Callable
        JIT-compiled ODE derivative function with signature (t, states, args) -> derivatives
    """
    @filter_jit
    def ode_derivative(t, states, args=None):
        _mus, _Ps = states
        d_mean, d_cov = ode_mean_and_cov(_mus, _Ps, process_diffusion, process_drift, sigma_points)
        return d_mean, d_cov

    return ode_derivative


def create_spgsf_update_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    sigma_points: SigmaPoints,
) -> Callable:
    """
    Create a standalone JIT-compiled SP-GSF Bayesian update function for cost analysis.

    Parameters
    ----------
    measurement_function : Callable[[Array], Array]
        The measurement function h(x).
    measurement_covariance : Array
        The measurement noise covariance matrix R.
    sigma_points : SigmaPoints
        Sigma points object for unscented transform.

    Returns
    -------
    Callable
        JIT-compiled update function with signature (mus, Ps, log_weights, meas) -> (mus, Ps, log_weights)
    """
    @filter_jit
    def spgsf_update(mus: Array, Ps: Array, log_weights: Array, meas: Array):
        mus_post, Ps_post, log_weight_evidence = bayesian_update(
            mus, Ps, meas, measurement_covariance, measurement_function, sigma_points
        )
        log_weights_post = normalize_log_weights(log_weight_evidence + log_weights)
        return mus_post, Ps_post, log_weights_post

    return spgsf_update
