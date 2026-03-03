from typing import Callable
import numpy as onp
import sympy as sp
import diffrax as dfx
from diffrax import AbstractSolver, Tsit5
from equinox import filter_jit
import jax.numpy as jnp
import jax
from jax._src.lax.control_flow import scan
from simulation.dynamical_system import DynamicalSystem
from cd_filtering.flows import FokkerPlanckFlowCholesky
from exponential_family.n_d_ef import NDExponentialFamily
import symbolic.n_d as nds
from utils.dfx_tqdm import WebTqdmProgressMeter
from jax.lax import stop_gradient
from jaxtyping import Array, Float, PyTree
from simulation.configs import ProjectionFilterConfig


def proj_error_norm_square_hist_scan(p:NDExponentialFamily,
                                     natural_statistics_symbolic: sp.Matrix,
                                     lc_jax: Callable[[Array, ], Array],
                                     dynamic: DynamicalSystem,
                                     theta_hist: Array,
                                     params_hist):
    """
    Compute the squared error norm history for the projection filter.

    This function calculates the squared error norm history for the projection filter by scanning through
    the history of natural parameters and bijection parameters. It computes the expected value of the
    squared forward Kullback-Leibler divergence and the backward Kolmogorov generator function.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution used for filtering.
    natural_statistics_symbolic : sp.Matrix
        Symbolic representation of the natural statistics.
    lc_jax : Callable[[Array], Array]
        Backward Kolmogorov generator function in JAX.
    dynamic : DynamicalSystem
        The dynamical system being simulated.
    theta_hist : Array
        History of natural parameters.
    params_hist : tuple
        History of bijection parameters.

    Returns
    -------
    Array
        History of squared error norms.
    """
    theta_symbolic = sp.Matrix(sp.symbols('theta1:{}'.format(len(natural_statistics_symbolic) + 1), real=True))
    a_jax_function = nds.square_fwd_klmgr_p_div_p_jax(natural_statistics_symbolic, dynamic.sde, theta_symbolic,
                                                      {})
    def _scanned_fun(_carry, _input):
        _theta, _params = _input

        @jax.jit
        def a_stat(_x):
            return a_jax_function(_x, _theta)

        _E_sqr_Lp_per_p = p.expected_value(a_stat, _theta, _params)
        _E_Lc = p.expected_value(lc_jax, _theta, _params)
        _fisher = p.fisher_metric(_theta, _params)

        _sqr_error_norm = 0.25 * (_E_sqr_Lp_per_p - _E_Lc @ (jnp.linalg.solve(_fisher, _E_Lc)))
        return _carry, _sqr_error_norm

    _, _sqr_error_norm_hist = scan(_scanned_fun, None, (theta_hist, params_hist))
    return _sqr_error_norm_hist

def get_T_matrices(ch: sp.Matrix, nat_stats: sp.Matrix) -> tuple:
    # it is assumed that the elements _ch_sym should be unique
    # get kronecker product
    _ch2_sym = sp.kronecker_product(ch, ch)

    _T_1 = onp.zeros((ch.shape[0], nat_stats.shape[0]), dtype=onp.int32)
    _T_2 = onp.zeros((_ch2_sym.shape[0], nat_stats.shape[0]), dtype=onp.int32)
    _repetition_T_2 = onp.zeros((nat_stats.shape[0],))

    for j, _stat in enumerate(nat_stats):
        for i, _ch_entry in enumerate(ch):
            if _stat == _ch_entry:
                _T_1[i, j] = 1
        for i, _ch_entry in enumerate(_ch2_sym):
            if _stat == _ch_entry:
                _T_2[i, j] = 1

    # normalization of _T_2
    _T_2 = _T_2 / _T_2.sum(axis=-1)[:, jnp.newaxis]
    return _T_1, _T_2

def get_theta_ell(_meas: Array, _args):
    _K1, theta_ell_0 = _args
    _theta_ell = -_K1 @ _meas + theta_ell_0
    return _theta_ell


def create_bayesian_update_for_cost_analysis(
    p: NDExponentialFamily,
    theta_ell: Callable[[Array, PyTree], Array],
    theta_ell_args: tuple,
    params_post_iter: int = 3,
) -> Callable:
    """
    Create a standalone JIT-compiled Bayesian update function for cost analysis.

    This function creates a standalone version of the Bayesian update step
    from the projection filter that can be compiled and analyzed independently
    to measure FLOPS per measurement update.

    The Bayesian update consists of:
    1. Computing theta_post = theta_pred - theta_ell(meas, theta_ell_args)
    2. Iteratively updating bijection parameters (params_post_iter times)

    Parameters
    ----------
    p : NDExponentialFamily
        Exponential family distribution object.
    theta_ell : Callable[[Array, PyTree], Array]
        Function computing the measurement update term.
    theta_ell_args : tuple
        Arguments for theta_ell function.
    params_post_iter : int, default=3
        Number of parameter update iterations.

    Returns
    -------
    Callable
        JIT-compiled Bayesian update function with signature
        (theta_pred, params_pred, meas) -> (theta_post, params_post)
    """
    @filter_jit
    def bayesian_update(theta_pred: Array, params_pred: tuple, meas: Array):
        theta_post = theta_pred - theta_ell(meas, theta_ell_args)
        params_post = params_pred

        def body_fn(i, params):
            eta_post = stop_gradient(p.natural_statistics_expectation(theta_post, params))
            return p.update_chol_bijection_params(eta_post, None, params)


        params_post = jax.lax.fori_loop(0, params_post_iter, body_fn, params_post)
        return theta_post, params_post

    return bayesian_update


@filter_jit
def cd_conj_proj_filter_chol(p: NDExponentialFamily,
                             cal_l_c_jax: Callable[[Array], Array],
                             theta_init: Array,
                             params_init: tuple[Array, Array, float],
                             time_sample: float,
                            #  dt_fp: float,
                             meas_record: Array,
                             theta_ell: Callable[[Array, PyTree], Array],
                             proj_filter_config: ProjectionFilterConfig,
                            #  theta_indices_for_bijection_params: tuple[Array, Array], #ProjectionFilterConfig
                            #  ode_solver: AbstractSolver = Tsit5(),#ProjectionFilterConfig
                            #  constant_step_size: bool = False,#ProjectionFilterConfig
                            #  rtol: float = 1e-3,#ProjectionFilterConfig
                            #  atol: float = 1e-8,#ProjectionFilterConfig
                            #  params_post_iter: int = 3,#ProjectionFilterConfig
                            #  theta_ell_args: tuple = None,#ProjectionFilterConfig
                            #  max_steps: int = int(1e12),#ProjectionFilterConfig
                             progress_bar_type: str = "",
                            #  save_fokker_planck_thetas: bool = False,#ProjectionFilterConfig
                            #  p_pid : float = 0.4,#ProjectionFilterConfig
                            #  i_pid : float = 1.,#ProjectionFilterConfig
                            #  d_pid : float = 0,#ProjectionFilterConfig
                            #  fisher_regularizer_initial_lambda: float = float(jnp.power(2.0,-32)),#ProjectionFilterConfig
                            #  fisher_regularizer_lambda_factor: int = 5,#ProjectionFilterConfig
                            #  fisher_regularizer_max_attempts: int = 15#ProjectionFilterConfig
                             ):

    """
    Applies a conjugate-duality projection filter using conjugate prior updates to a sequence of measurements.

    The filter consists of:
    1) A prediction step where the Fokker-Planck flow evolves the distribution
    2) A Bayesian update step assuming conjugate prior structure

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution used for filtering.
    cal_l_c_jax : Callable[[Array], Array]
        Calibration function for the Fokker-Planck flow.
    theta_init : Array
        Initial natural parameters.
    params_init : tuple[Array, Array, float]
        Initial parameters for the distribution bijection.
    time_sample : float
        Time interval between measurements.
    dt_fp : float
        Initial time step for solving Fokker--Planck flow..
    meas_record : Array
        Sequence of measurements to filter.
    theta_ell : Callable[[Array, PyTree], Array]
        Function computing natural parameter update from measurement.
    theta_indices_for_bijection_params : Array
        Indices mapping natural parameters to bijection parameters.
    ode_solver : AbstractSolver, optional
        ODE solver to use, defaults to Tsit5.
    constant_step_size : bool, optional
        Whether to use constant step size, defaults to False.
    rtol : float, optional
        Relative tolerance for ODE solver, defaults to 1e-3.
    atol : float, optional
        Absolute tolerance for ODE solver, defaults to 1e-8.
    params_post_iter : int, optional
        Number of iterations for posterior parameter updates, defaults to 3.
    theta_ell_args : tuple, optional
        Additional arguments passed to theta_ell function.
    max_steps : int, optional
        Maximum number of steps for ODE solver, defaults to 1e12.
    progress_bar_type : str, optional
        Whether to show progress bar during computation, defaults to no progress bar. The other options are text, and web.
    save_fokker_planck_thetas : bool, optional
        Whether to save intermediate Fokker-Planck theta values, defaults to False.
    p_pid : float, optional
        Proportional coefficient for the PID controller, defaults to 0.4.
    i_pid : float, optional
        Integral coefficient for the PID controller, defaults to 1.0.
    d_pid : float, optional
        Derivative coefficient for the PID controller, defaults to 0.
    Returns
    -------
    tuple
        Results containing posterior parameters, learning rates and predicted parameters at each step.

    Notes
    -----
    tip "Choosing PID coefficients"

        This controller can be reduced to any special case (e.g. just a PI controller,
        or just an I controller) by setting `pcoeff`, `icoeff` or `dcoeff` to zero
        as appropriate.

        For smoothly-varying (i.e. easy to solve) problems then an I controller, or a
        PI controller with `icoeff=1`, will often be most efficient.
        ```python
        PIDController(pcoeff=0,   icoeff=1, dcoeff=0)  # default coefficients
        PIDController(pcoeff=0.4, icoeff=1, dcoeff=0)
        ```

        For moderate difficulty problems that may have an error estimate that does
        not vary smoothly, then a less sensitive controller will often do well. (This
        includes many mildly stiff problems.) Several different coefficients are
        suggested in the literature, e.g.
        ```python
        PIDController(pcoeff=0.4, icoeff=0.3, dcoeff=0)
        PIDController(pcoeff=0.3, icoeff=0.3, dcoeff=0)
        PIDController(pcoeff=0.2, icoeff=0.4, dcoeff=0)
    """

    an_fp_flow = FokkerPlanckFlowCholesky(p,
                                          cal_l_c_jax,
                                        #   theta_indices_for_bijection_params,
                                          params_init[-1],
                                        #   fisher_regularizer_initial_lambda,
                                        #   fisher_regularizer_lambda_factor,
                                        #   fisher_regularizer_max_attempts,
                                        proj_filter_config
                                          )

    ode_term = dfx.ODETerm(an_fp_flow)

    if proj_filter_config.constant_step_size:
        stepsize_controller = dfx.ConstantStepSize()
    else:
        stepsize_controller = dfx.PIDController(pcoeff=proj_filter_config.p_pid, icoeff=proj_filter_config.i_pid, dcoeff=proj_filter_config.d_pid,
                                                rtol=proj_filter_config.rtol, atol=proj_filter_config.atol)

    progress_bar = dfx.NoProgressMeter()
    if progress_bar_type.lower() == "web":
        progress_bar = WebTqdmProgressMeter()
    elif progress_bar_type.lower() == "text":
        progress_bar = dfx.TqdmProgressMeter()

    # use this if only the last value of theta_pred will be saved
    def scanned_fun_default(_carry, _input):
        theta, params, t = _carry
        meas = _input
        saveat = dfx.SaveAt(t1=True)

        fp_state = theta, params[0], params[1]


        fp_sol = dfx.diffeqsolve(ode_term, proj_filter_config.ode_solver, t, t + time_sample, dt0=proj_filter_config.dt_fp,
                                 y0=fp_state, saveat=saveat,
                                 stepsize_controller=stepsize_controller,
                                 max_steps=proj_filter_config.ode_max_steps,
                                 progress_meter=progress_bar,
                                 throw=True)
        # jax.debug.print('Fokker--Planck step finished!')

        theta_pred, params_0_pred, params_1_pred = fp_sol.ys
        params_pred = (params_0_pred.squeeze(), params_1_pred.squeeze(), params_init[-1])
        theta_pred = theta_pred.squeeze()
        # jax.debug.print("Fokker--Planck step is completed. mu-params = {}", params_pred[0])

        theta_post = theta_pred - theta_ell(meas, proj_filter_config.theta_ell_args)
        params_post = params_pred

        # compute params_post for mmt_iter times
        for i in range(proj_filter_config.mmt_iter):
            # since we do not want to break the relation between old params and new params
            # we freeze eta_post
            eta_post = stop_gradient(p.natural_statistics_expectation(theta_post, params_post))
            params_post = p.update_chol_bijection_params(eta_post, None, params_post)

        _carry = (theta_post, params_post, t + time_sample)
        _output = (theta_post, params_post, 0., theta_pred, params_pred, fp_sol)
        return _carry, _output

    # use this if all values of theta_pred should be saved
    def scanned_fun_fp(_carry, _input):
        theta, params, t = _carry
        meas = _input
        fp_state = theta, params[0], params[1]
        ts = jnp.linspace(t, t + time_sample, int(time_sample/proj_filter_config.dt))
        saveat = dfx.SaveAt(ts=ts)
        fp_sol = dfx.diffeqsolve(ode_term, proj_filter_config.ode_solver, t, t + time_sample, dt0=proj_filter_config.dt,
                                 y0=fp_state, saveat=saveat,
                                 stepsize_controller=stepsize_controller, max_steps=proj_filter_config.ode_max_steps,
                                 progress_meter=progress_bar,
                                 throw=True)
        # jax.debug.print('Fokker--Planck step finished!')

        theta_pred, params_0_pred, params_1_pred = fp_sol.ys[0][-1], fp_sol.ys[1][-1], fp_sol.ys[2][-1]
        params_pred = (params_0_pred.squeeze(), params_1_pred.squeeze(), params_init[-1])
        theta_pred = theta_pred.squeeze()
        jax.debug.print("Fokker--Planck step is completed. mu-params = {}", params_pred[0])

        theta_post = theta_pred - theta_ell(meas, proj_filter_config.theta_ell_args)
        params_post = params_pred



        # compute params_post for params_post_iter times
        for i in range(proj_filter_config.mmt_iter):
            # since we do not want to break the relation between old params and new params
            # we freeze eta_post
            eta_post = stop_gradient(p.natural_statistics_expectation(theta_post, params_post))
            params_post = p.update_chol_bijection_params(eta_post,
                                                            None,
                                                            params_post)

        _carry = (theta_post, params_post, t + time_sample)
        # also include the whole fp_sol for analysis
        _output = (theta_post, params_post, 0., fp_sol.ys, fp_sol.ts, fp_sol)
        return _carry, _output

    scanned_fun = scanned_fun_default
    if proj_filter_config.save_fokker_planck_thetas:
        scanned_fun = scanned_fun_fp

    _, results = scan(scanned_fun, (theta_init, params_init, 0.), xs=meas_record)
    return results
