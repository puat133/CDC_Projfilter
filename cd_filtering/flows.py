
from typing import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
from diffrax import AbstractSolver
from equinox import Module, filter_jit
from jax import value_and_grad
from jax.lax import stop_gradient
import utils.natural_gradient as ng

import cd_filtering.bayesian_update as bu
from exponential_family.n_d_ef import NDExponentialFamily
from utils.density_manipulations import bijection_parameters_time_derivative, \
    cholesky_bijection_parameters_time_derivative
from jaxtyping import Array, Float
from simulation.configs import ProjectionFilterConfig


@filter_jit
def theta_ode(p, theta, params, cal_l_c_jax):
    fisher = p.fisher_metric(theta, params)
    euclidian_grad = p.expected_value(cal_l_c_jax, theta, params)
    return jnp.linalg.solve(fisher, euclidian_grad), euclidian_grad


jacobian_of_theta_ode = jax.jacobian(theta_ode, 1)
class FokkerPlanckFlowCholesky(Module):
    p: NDExponentialFamily
    cal_l_c_jax: Callable[[Array], Array]
    scale_params: Float
    proj_filter_config: ProjectionFilterConfig
    """
    Flow that computes the Fokker-Planck forward equation for exponential families.
    Uses the cholesky of covariance as bijection parameters instead of the original covariance.

    Parameters
    ----------
    p : NDExponentialFamily
        Exponential family distribution object.
    cal_l_c_jax : Callable[Array -> Array]
        Callable function used in expected value computation
    theta_indices_for_bijection_params : Array
        Indices mapping theta parameters to bijection parameters
    scale_params : Float
        Scale parameter for the distribution

    Methods
    -------
    __call__(t, states, args) -> tuple
        Computes derivatives for the flow equations.
        Returns tuple of (d_theta_dt, d_mu_dt, d_cov_dt)
    """
    def __call__(self,
                 t,
                 states,
                 args
                 ) -> tuple:
        theta, mu_params, chol_params = states

        params = (mu_params, chol_params, self.scale_params)


        fisher = self.p.fisher_metric(theta, params)

        # since `euclidean_grad` is used to update theta and parameters, we stop_gradient from `euclidean_grad`
        # we do not want to see relation between old theta and new theta, and old parameters and new parameters
        euclidian_grad = stop_gradient(self.p.expected_value(self.cal_l_c_jax, theta, params))
        d_theta_dt = ng.eigenvalue_truncation_natural_gradient(fisher,
                                                             euclidian_grad,
                                                             self.proj_filter_config.min_fisher_ev,
                                                             self.proj_filter_config.max_d_theta_dt_norm)

        jax.debug.print('t = {:.6e}, norm of d_eta_dt = {:.6e}, norm of d_theta_dt = {:.6e}, norm of actual d_theta_dt = {:.6e}', t, 
                        jnp.linalg.norm(euclidian_grad), 
                        jnp.linalg.norm(d_theta_dt), 
                        jnp.linalg.norm(jnp.linalg.solve(fisher, euclidian_grad)))
        d_mu_dt, d_S_dt = cholesky_bijection_parameters_time_derivative(euclidian_grad, params, 
                                                                        self.proj_filter_config.theta_indices_for_bijection_params)

        return d_theta_dt, d_mu_dt, d_S_dt


def fokker_planck_flow_cholesky(t:float, states:tuple, args: tuple) -> tuple:
    
    scale_params, p, cal_l_c_jax, theta_indices_for_bijection_params = args

    theta, mu_params, chol_params = states

    params = (mu_params, chol_params, scale_params)

    fisher = p.fisher_metric(theta, params)

    # since `euclidean_grad` is used to update theta and parameters, we stop_gradient from `euclidean_grad`
    # we do not want to see relation between old theta and new theta, and old parameters and new parameters
    euclidian_grad = stop_gradient(p.expected_value(cal_l_c_jax, theta, params))
    d_theta_dt = jnp.linalg.solve(fisher, euclidian_grad)
    d_mu_dt, d_S_dt = cholesky_bijection_parameters_time_derivative(euclidian_grad, params,
                                                                    theta_indices_for_bijection_params)

    return d_theta_dt, d_mu_dt, d_S_dt



class ConvexConjugateFlowCholesky(Module):
    p: NDExponentialFamily
    theta_init: Array
    eta: Array
    theta_indices_for_bijection_params: tuple[ Array, Array]
    scale_params: Float
    psi_theta_init: Float
    alpha: Float
    beta: Float
    """
    Flow that computes the convex conjugate using gradient flow with Cholesky parameterization.

    Parameters
    ----------
    p : NDExponentialFamily
        Exponential family distribution object.
    theta_init : Array
        Initial theta parameters
    eta : Array
        Eta parameters for convex conjugate
    theta_indices_for_bijection_params : Array
        Indices mapping theta parameters to bijection parameters
    scale_params : Float
        Scale parameter for the distribution
    psi_theta_init : Float
        Initial value of psi(theta)
    alpha : Float
        Learning rate adaptation parameter
    beta : Float
        Learning rate adaptation parameter

    Methods
    -------
    __call__(t, states, args) -> tuple
        Computes derivatives for the convex conjugate flow equations.
        Returns tuple of (d_theta_dt, d_mu_dt, d_cov_dt, dlearn_rate_dt)
    """

    def __call__(self,
                 t,
                 states,
                 args
                 ) -> tuple:
        theta, mu_params, chol_sigma_params, learn_rate = states
        params = (mu_params, chol_sigma_params, self.scale_params)

        fisher = self.p.fisher_metric(theta, params)
        convex_conjugate_args = (self.p, theta, self.eta, params)
        # since `euclidean_grad` is used to update theta and parameters, we stop_gradient from `euclidean_grad`
        # we do not want to see relation between old theta and new theta, and old parameters and new parameters
        objective, euclidian_grad = stop_gradient(value_and_grad(bu.convex_conjugate, 1)
                                                  (*convex_conjugate_args))

        # since this is to obtain a supremum, then there is no minus
        d_theta_dt = 4 * learn_rate * jnp.linalg.solve(fisher, euclidian_grad)

        d_mu_dt, d_S_dt = cholesky_bijection_parameters_time_derivative(
            4 * learn_rate * euclidian_grad,
            params,
            self.theta_indices_for_bijection_params)

        dlearn_rate_dt = self.alpha * learn_rate * (self.beta * objective - learn_rate)
        return d_theta_dt, d_mu_dt, d_S_dt, dlearn_rate_dt


@filter_jit
def propagate_convex_conjugate_cholesky_flow(p: NDExponentialFamily,
                                            theta_init: Array,
                                            eta: Array,
                                            params_init: tuple[Array, Array, float],
                                            theta_indices_for_bijection_params: tuple[ Array, Array],
                                            psi_theta_init: Float,
                                            learn_rate_init: Float,
                                            alpha: Float,
                                            beta: Float,
                                            ode_solver: AbstractSolver,
                                            constant_step_size: bool = False,
                                            rtol: Float = 1e-3,
                                            atol: Float = 1e-8,
                                            t1=jnp.inf,
                                            dt_bayes: Float = 1e-3,
                                            max_steps: int = int(1e12)) -> dfx.Solution:
    """
    Propagate the convex conjugate flow equations with Cholesky parameterization.

    Parameters
    ----------
    p : NDExponentialFamily
        Exponential family distribution object
    theta_init : Array
        Initial theta parameters
    eta : Array
        Eta parameters for convex conjugate
    params_init : tuple[Array, Array, float]
        Initial distribution parameters (mu, cov, scale)
    theta_indices_for_bijection_params : Array
        Indices mapping theta parameters to bijection parameters
    psi_theta_init : Float
        Initial value of psi(theta)
    learn_rate_init : Float
        Initial learning rate
    alpha : Float
        Learning rate adaptation parameter
    beta : Float
        Learning rate adaptation parameter
    ode_solver : AbstractSolver
        ODE solver object from diffrax
    constant_step_size : bool, default=False
        If True, use constant step size instead of adaptive
    rtol : Float, default=1e-3
        Relative tolerance for adaptive stepping
    atol : Float, default=1e-8
        Absolute tolerance for adaptive stepping
    t1 : Float, default=jnp.inf
        Final time
    dt_bayes : Float, default=1e-3
        Time step for constant step size
    max_steps : int, default=1e12
        Maximum number of integration steps

    Returns
    -------
    dfx.Solution
        Solution object containing integrated states
    """
    t0 = 0

    a_conjugate_flow = ConvexConjugateFlowCholesky(p,
                                           theta_init,
                                           eta,
                                           theta_indices_for_bijection_params,
                                           params_init[-1],  # scale_param
                                           psi_theta_init,
                                           alpha,
                                           beta)
    terms = dfx.ODETerm(a_conjugate_flow)

    initial_states = (theta_init, params_init[0], params_init[1], learn_rate_init)
    if constant_step_size:
        stepsize_controller = dfx.ConstantStepSize()
        a_sol = dfx.diffeqsolve(
            terms,
            ode_solver,
            t0,
            t1,
            dt0=dt_bayes,
            y0=initial_states,
            stepsize_controller=stepsize_controller
        )
    else:
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)
        dt_bayes = None

        a_sol = dfx.diffeqsolve(
            terms,
            ode_solver,
            t0,
            t1,
            dt0=dt_bayes,
            y0=initial_states,
            stepsize_controller=stepsize_controller,
            event=dfx.Event(dfx.steady_state_event()),
            max_steps=max_steps
        )

    return a_sol



def create_fokker_planck_flow_cholesky_for_cost_analysis(
    p: NDExponentialFamily,
    cal_l_c_jax: Callable[[Array], Array],
    scale_params: Float,
    proj_filt_config: ProjectionFilterConfig
) -> Callable:
    """
    Create a standalone JIT-compiled Cholesky flow derivative function for cost analysis.

    This function creates a standalone version of the FokkerPlanckFlowCholesky.__call__
    that can be compiled and analyzed independently to measure FLOPS per ODE step.

    Parameters
    ----------
    p : NDExponentialFamily
        Exponential family distribution object.
    cal_l_c_jax : Callable[[Array], Array]
        Callable function used in expected value computation.
    theta_indices_for_bijection_params : tuple[Array, Array]
        Indices mapping theta parameters to bijection parameters.
    scale_params : Float
        Scale parameter for the distribution.
    min_fisher_ev : float, default=1e-1
        Minimum eigenvalue threshold for eigenvalue truncation.
    max_d_theta_dt_norm : float, default=1e2
        Maximum norm for d_theta_dt.

    Returns
    -------
    Callable
        JIT-compiled flow derivative function with signature (t, states, args) -> derivatives
    """
    @filter_jit
    def flow_derivative(t, states, args=None):
        theta, mu_params, chol_params = states
        params = (mu_params, chol_params, scale_params)

        fisher = p.fisher_metric(theta, params)
        euclidian_grad = stop_gradient(p.expected_value(cal_l_c_jax, theta, params))

        # d_theta_dt = regularized_natural_gradient(fisher, 
        #                                           euclidian_grad, 
        #                                           proj_filt_config.fisher_regularizer_initial_lambda,
        #                                           proj_filt_config.fisher_regularizer_lambda_factor,
        #                                           proj_filt_config.fisher_regularizer_max_attempts)
        d_theta_dt = ng.eigenvalue_truncation_natural_gradient(fisher, 
                                                             euclidian_grad,
                                                             proj_filt_config.min_fisher_ev,
                                                             proj_filt_config.max_d_theta_dt_norm) # use default parameters here as the FLOPs will be the same

        d_mu_dt, d_S_dt = cholesky_bijection_parameters_time_derivative(
            euclidian_grad, params,
            proj_filt_config.theta_indices_for_bijection_params)

        return d_theta_dt, d_mu_dt, d_S_dt

    return flow_derivative
