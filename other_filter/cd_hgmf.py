"""
Continuous-Discrete Homotopic Gaussian Mixture Filter (HGMF).

Implements the homotopic update of Craft & DeMars,
"Homotopic Gaussian Mixture Filtering for Applied Bayesian Inference",
IEEE TAC, Vol. 70, No. 7, July 2025, pp. 4608-4623.

Prediction stage: continuous-time Gaussian flow per component (sigma-point UT/GHQ),
reusing SigmaPointKalmanFlow from cd_sp_kf.

Update stage: per-component ODE in homotopy variable s in [0, 1] (or
log-homotopy variable tau = log(s + eps) -- Corollary 5 in the paper),
parameterized by Taylor-2 statistics of the log-likelihood about each
component's mean (eqs. 10-11 and 17a-c of the paper).

For each measurement z, every component i integrates:
    dm^{(i)}/ds        = P^{(i)}(s) . g^{(i)}(s)
    dP^{(i)}/ds        = P^{(i)}(s) . L^{(i)}(s) . P^{(i)}(s)
    d log w_tilde^{(i)}/ds = ell^{(i)}(s) + 0.5 * tr(L^{(i)}(s) . P^{(i)}(s))

with (for a Gaussian observation z = h(x) + nu, nu ~ N(0, R)):
    h_eval^{(i)} = h(m^{(i)}(s)),  H^{(i)} = dh/dx |_{m^{(i)}(s)}
    g^{(i)} = H^{(i)T} R^{-1} (z - h_eval^{(i)})
    L^{(i)} = -H^{(i)T} R^{-1} H^{(i)}                       # drop measurement-Hessian term
    ell^{(i)} = -0.5 (z - h_eval^{(i)})^T R^{-1} (z - h_eval^{(i)})  # constant -0.5 log|2 pi R| cancels in renormalisation

The log-unnormalised weights are renormalised via logsumexp after each
measurement (Corollary 6).
"""
from typing import Callable

import jax
import jax.numpy as jnp
import diffrax as dfx
from diffrax import AbstractSolver
from equinox import Module, filter_jit
from jax.lax import scan
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, PyTree

from other_filter.cd_sp_kf import SigmaPointKalmanFlow
from sigma_points.sigma_points import SigmaPoints


class HomotopicGaussianMeasurement(Module):
    """Log-likelihood / gradient / Hessian utilities for a nonlinear Gaussian
    measurement z = h(x) + nu, nu ~ N(0, R), evaluated at a component mean
    (Taylor-2 simplification of eqs. 17a-c)."""

    h_fun: Callable[[Array], Array]
    h_jac: Callable[[Array], Array]
    R_inv: Array

    def __init__(self,
                 measurement_function: Callable[[Array], Array],
                 measurement_covariance: Array):
        self.h_fun = measurement_function
        self.h_jac = jax.jacobian(measurement_function)
        self.R_inv = jnp.linalg.inv(measurement_covariance)

    def log_p_grad_hess(self, m: Array, z: Array) -> tuple[Array, Array, Array]:
        """Return (log p, gradient, Hessian) at x = m given measurement z.

        The constant -0.5 log|2 pi R| is dropped because it cancels in the
        homotopic weight renormalisation.
        """
        h_eval = self.h_fun(m)
        H = self.h_jac(m)
        innov = z - h_eval
        Rinv_innov = self.R_inv @ innov
        log_p = -0.5 * innov @ Rinv_innov
        grad = H.T @ Rinv_innov
        hess = -H.T @ self.R_inv @ H
        return log_p, grad, hess


class HomotopicCorrectionFlow(Module):
    """Per-component homotopic ODE in s (or tau if log-homotopy).

    State (pytree) at each call: (means, covs, log_w_tilde) with leading axis
    of size N (number of mixture components).
    """

    lik: HomotopicGaussianMeasurement
    use_log_homotopy: bool
    log_homotopy_eps: float

    def __call__(self, s_or_tau, states, args) -> tuple:
        z = args  # frozen across the s-integration
        means, covs, _ = states

        def per_component(m_i, P_i):
            _, g, L = self.lik.log_p_grad_hess(m_i, z)
            # ell evaluated at the (potentially updated) component mean; the
            # log-likelihood and the (g, L) are recomputed here to avoid
            # discarding components that we just paid the price to compute.
            innov = z - self.lik.h_fun(m_i)
            ell = -0.5 * innov @ (self.lik.R_inv @ innov)

            dm = P_i @ g
            P_L = P_i @ L
            dP = P_L @ P_i
            # d log w_tilde/ds = ell + 0.5 tr(L P) = ell + 0.5 tr(P L)
            dlw = ell + 0.5 * jnp.trace(P_L)
            return dm, dP, dlw

        dmeans, dcovs, dlog_w_tilde = jax.vmap(per_component)(means, covs)

        if self.use_log_homotopy:
            # ds/dtau = exp(tau) = s + eps
            scale = jnp.exp(s_or_tau)
            dmeans = scale * dmeans
            dcovs = scale * dcovs
            dlog_w_tilde = scale * dlog_w_tilde

        return dmeans, dcovs, dlog_w_tilde


@filter_jit
def cd_hgmf(
    process_drift: Callable[[Float, Array, PyTree], Array],
    process_diffusion: Callable[[Float, Array, PyTree], Array],
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    time_sample: Float,
    means_init: Array,
    covs_init: Array,
    log_weights_init: Array,
    meas_record: Array,
    sigma_points: SigmaPoints,
    pred_ode_solver: AbstractSolver = dfx.Tsit5(),
    hom_ode_solver: AbstractSolver = dfx.Tsit5(),
    constant_step_size: bool = False,
    rtol_pred: Float = 1e-3,
    atol_pred: Float = 1e-6,
    rtol_hom: Float = 1e-3,
    atol_hom: Float = 1e-6,
    dt_pred: Float = 1e-3,
    dt_hom: Float = 1e-2,
    use_log_homotopy: bool = True,
    log_homotopy_eps: Float = 1e-3,
):
    """Continuous-discrete Homotopic Gaussian Mixture Filter.

    Parameters
    ----------
    process_drift, process_diffusion
        SDE drift f(t, x, args) and diffusion g(t, x, args).
    measurement_function, measurement_covariance
        h(x) and R for z = h(x) + nu, nu ~ N(0, R).
    time_sample
        Physical time interval between measurements.
    means_init, covs_init, log_weights_init
        Initial GM parameters with shapes (N, n), (N, n, n), (N,).
    meas_record
        Sequence of measurements, shape (n_meas, dim_meas).
    sigma_points
        Sigma-point object used for the prediction Gaussian flow.
    pred_ode_solver, hom_ode_solver
        Diffrax solvers for prediction (over physical time) and homotopic
        correction (over s or tau).
    rtol_pred, atol_pred, rtol_hom, atol_hom
        PID adaptive-step tolerances for the two ODE integrations.
    dt_pred, dt_hom
        Initial step sizes.
    use_log_homotopy
        If True, integrate over tau = log(s + log_homotopy_eps) instead of s
        directly to ameliorate the stiffness near s = 0 (paper's Corollary 5).
    log_homotopy_eps
        eps in tau = log(s + eps); ignored when use_log_homotopy is False.

    Returns
    -------
    (means_hist, covs_hist, log_weights_hist, pred_sol, hom_sol)

    - means_hist : (n_meas, N, n)
    - covs_hist  : (n_meas, N, n, n)
    - log_weights_hist : (n_meas, N)
    - pred_sol : diffrax Solution from the prediction stage (stats for FLOPS).
    - hom_sol  : diffrax Solution from the homotopic stage   (stats for FLOPS).
    """
    pred_term = dfx.ODETerm(SigmaPointKalmanFlow(process_drift, process_diffusion, sigma_points))

    lik = HomotopicGaussianMeasurement(measurement_function, measurement_covariance)
    hom_flow = HomotopicCorrectionFlow(lik, use_log_homotopy, log_homotopy_eps)
    hom_term = dfx.ODETerm(hom_flow)

    saveat_t1 = dfx.SaveAt(t1=True)

    if constant_step_size:
        pred_controller = dfx.ConstantStepSize()
        hom_controller = dfx.ConstantStepSize()
    else:
        pred_controller = dfx.PIDController(rtol=rtol_pred, atol=atol_pred)
        hom_controller = dfx.PIDController(rtol=rtol_hom, atol=atol_hom)

    if use_log_homotopy:
        s0 = jnp.log(log_homotopy_eps)
        s1 = jnp.log(1.0 + log_homotopy_eps)
    else:
        s0 = 0.0
        s1 = 1.0

    def scanned_fun(_carry, _input):
        _t0, _means, _covs, _log_w = _carry
        _meas = _input

        # --- Prediction over physical time [t0, t0 + dt] ---
        pred_sol = dfx.diffeqsolve(
            pred_term,
            pred_ode_solver,
            _t0,
            _t0 + time_sample,
            dt0=dt_pred,
            y0=(_means, _covs),
            saveat=saveat_t1,
            stepsize_controller=pred_controller,
        )
        _means_pred, _covs_pred = pred_sol.ys
        _means_pred = _means_pred.squeeze(0)
        _covs_pred = _covs_pred.squeeze(0)

        # Symmetrise covariances for numerical safety.
        _covs_pred = 0.5 * (_covs_pred + jnp.swapaxes(_covs_pred, -1, -2))

        # --- Homotopic correction over s in [0, 1] (or tau in log-homotopy) ---
        hom_sol = dfx.diffeqsolve(
            hom_term,
            hom_ode_solver,
            s0,
            s1,
            dt0=dt_hom,
            y0=(_means_pred, _covs_pred, _log_w),
            args=_meas,
            saveat=saveat_t1,
            stepsize_controller=hom_controller,
        )
        _means_post, _covs_post, _log_w_tilde = hom_sol.ys
        _means_post = _means_post.squeeze(0)
        _covs_post = _covs_post.squeeze(0)
        _log_w_tilde = _log_w_tilde.squeeze(0)

        # Symmetrise + renormalise.
        _covs_post = 0.5 * (_covs_post + jnp.swapaxes(_covs_post, -1, -2))
        _log_w_post = _log_w_tilde - logsumexp(_log_w_tilde)

        _carry = (_t0 + time_sample, _means_post, _covs_post, _log_w_post)
        return _carry, (_means_post, _covs_post, _log_w_post, pred_sol, hom_sol)

    init_carry = (0.0, means_init, covs_init, log_weights_init)
    _, results = scan(scanned_fun, init_carry, xs=meas_record)
    return results


# ---------------------------------------------------------------------------
# Cost-analysis factories
# ---------------------------------------------------------------------------


def create_hgmf_pred_ode_for_cost_analysis(
    process_drift: Callable[[Float, Array, PyTree], Array],
    process_diffusion: Callable[[Float, Array, PyTree], Array],
    sigma_points: SigmaPoints,
) -> Callable:
    """JIT-compiled prediction ODE derivative for FLOPS cost analysis."""
    flow = SigmaPointKalmanFlow(process_drift, process_diffusion, sigma_points)

    @filter_jit
    def pred_derivative(t, states, args=None):
        return flow(t, states, args)

    return pred_derivative


def create_hgmf_hom_ode_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    use_log_homotopy: bool = True,
    log_homotopy_eps: Float = 1e-3,
) -> Callable:
    """JIT-compiled homotopic ODE derivative for FLOPS cost analysis."""
    lik = HomotopicGaussianMeasurement(measurement_function, measurement_covariance)
    flow = HomotopicCorrectionFlow(lik, use_log_homotopy, log_homotopy_eps)

    @filter_jit
    def hom_derivative(s, states, meas):
        return flow(s, states, meas)

    return hom_derivative


def create_hgmf_update_for_cost_analysis(
    measurement_function: Callable[[Array], Array],
    measurement_covariance: Array,
    pred_ode_solver: AbstractSolver = dfx.Tsit5(),
    hom_ode_solver: AbstractSolver = dfx.Tsit5(),
    rtol_hom: Float = 1e-3,
    atol_hom: Float = 1e-6,
    dt_hom: Float = 1e-2,
    use_log_homotopy: bool = True,
    log_homotopy_eps: Float = 1e-3,
) -> Callable:
    """JIT-compiled full homotopic correction (diffeqsolve over s) for FLOPS analysis."""
    lik = HomotopicGaussianMeasurement(measurement_function, measurement_covariance)
    hom_flow = HomotopicCorrectionFlow(lik, use_log_homotopy, log_homotopy_eps)
    hom_term = dfx.ODETerm(hom_flow)
    hom_controller = dfx.PIDController(rtol=rtol_hom, atol=atol_hom)
    saveat_t1 = dfx.SaveAt(t1=True)

    if use_log_homotopy:
        s0 = jnp.log(log_homotopy_eps)
        s1 = jnp.log(1.0 + log_homotopy_eps)
    else:
        s0 = 0.0
        s1 = 1.0

    @filter_jit
    def hgmf_update(means: Array, covs: Array, log_weights: Array, meas: Array):
        hom_sol = dfx.diffeqsolve(
            hom_term,
            hom_ode_solver,
            s0,
            s1,
            dt0=dt_hom,
            y0=(means, covs, log_weights),
            args=meas,
            saveat=saveat_t1,
            stepsize_controller=hom_controller,
        )
        means_post, covs_post, log_w_tilde = hom_sol.ys
        means_post = means_post.squeeze(0)
        covs_post = covs_post.squeeze(0)
        log_w_tilde = log_w_tilde.squeeze(0)
        covs_post = 0.5 * (covs_post + jnp.swapaxes(covs_post, -1, -2))
        log_w_post = log_w_tilde - logsumexp(log_w_tilde)
        return means_post, covs_post, log_w_post

    return hgmf_update
