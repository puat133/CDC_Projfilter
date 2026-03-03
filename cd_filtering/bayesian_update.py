from functools import partial
from typing import Callable

from equinox import filter_jit
from jax import jit, hessian, jacrev, numpy as jnp
from jax.lax import scan

from exponential_family.n_d_ef import NDExponentialFamily
from jaxtyping import Array, Float

@partial(jnp.vectorize, signature='(d)->()', excluded=(1, 2, 3, 4, 5))
def inner_fun(x: Array,
              p: NDExponentialFamily,
              neg_log_likelihood: Callable[[Array, Array], Array],
              theta: Array,
              theta_ast: Array,
              meas: Array):
    """
    Computes a value involving natural statistics, negative log likelihood, and parameter differences.

    Parameters
    ----------
    x : Array
        Input data point.
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta : Array
        Parameter vector for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.

    Returns
    -------
    float
        Computed value based on natural statistics and negative log likelihood.
    """
    return 0.5 * (p.natural_statistics(x) @ (theta_ast - theta) - neg_log_likelihood(x, meas))


@filter_jit
def A_half_theta(p: NDExponentialFamily,
                 neg_log_likelihood: Callable[[Array, Array], Array],
                 theta: Array,
                 theta_ast: Array,
                 meas: Array,
                 params: tuple[Array, Array, float],
                 psi_theta_ast: Float,
                 z_theta_ast: Float):
    """
    Computes a scaling factor A(theta) for the divergence measure.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta : Array
        Parameter vector for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params : tuple[Array, Array, float]
        Parameters for the distribution (e.g., moments, variance).
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    float
        Scaling factor A(theta).
    """
    psi_theta = p.log_partition(theta, params)

    def a_statistic(x):
        return jnp.exp(inner_fun(x, p, neg_log_likelihood, theta, theta_ast, meas)
                       + 0.5 * (psi_theta - psi_theta_ast - z_theta_ast))

    return p.expected_value(a_statistic, theta, params)


@filter_jit
def divergence_half(p: NDExponentialFamily,
                    neg_log_likelihood: Callable[[Array, Array], Array],
                    theta: Array,
                    theta_ast: Array,
                    meas: Array,
                    params: tuple[Array, Array, float],
                    psi_theta_ast: Float,
                    z_theta_ast: Float):
    """
    Computes the half-divergence between two parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta : Array
        Parameter vector for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params : tuple[Array, Array, float]
        Parameters for the distribution (e.g., moments, variance).
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    float
        Computed half-divergence.
    """
    psi_theta = p.log_partition(theta, params)

    def log_fun(x):
        return inner_fun(x, p, neg_log_likelihood, theta, theta_ast, meas)

    return -2 * p.log_expected_value(log_fun, theta, params) + (psi_theta_ast + z_theta_ast - psi_theta)


@filter_jit
def divergence_half_expoenential(psi_theta_1: Float,
                                 psi_theta_2: Float,
                                 psi_theta_half: Float) -> float:
    """
    Computes the half divergence for exponential families based on given log-partition values.

    Parameters
    ----------
    psi_theta_1 : Float
        Log-partition function evaluated at theta_1.
    psi_theta_2 : Float
        Log-partition function evaluated at theta_2.
    psi_theta_half : Float
        Log-partition function evaluated at an intermediate value.

    Returns
    -------
    float
        Computed half divergence.
    """
    return -4 * (psi_theta_half - 0.5 * (psi_theta_1 + psi_theta_2))


@filter_jit
def kl_divergence(p: NDExponentialFamily,
                  neg_log_likelihood: Callable[[Array, Array], Array],
                  theta: Array,
                  theta_ast: Array,
                  meas: Array,
                  params: tuple,
                  params_ast: tuple,
                  psi_theta_ast: Float,
                  z_theta_ast: Float):
    """
    Computes the Kullback-Leibler (KL) divergence between two parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta : Array
        Parameter vector for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params : tuple
        Parameters for the distribution.
    params_ast : tuple
        Parameters for the reference distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    float
        Computed KL divergence.
    """
    psi_theta = p.log_partition(theta, params)

    def a_statistic(_x):
        _ell_y = neg_log_likelihood(_x, meas)
        return jnp.exp(-_ell_y - z_theta_ast) * (p.natural_statistics(_x) @ (theta_ast - theta) - (
                psi_theta_ast - psi_theta) - _ell_y - z_theta_ast)

    return p.expected_value(a_statistic, theta_ast, params_ast)


@filter_jit
def natural_statistic_alpha(p: NDExponentialFamily,
                            neg_log_likelihood: Callable[[Array, Array], Array],
                            theta: Array,
                            theta_ast: Array,
                            meas: Array,
                            params: tuple[Array, Array, float],
                            psi_theta_ast: Float,
                            z_theta_ast: Float):
    """
    Computes the natural statistic value for alpha-divergence.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta : Array
        Parameter vector for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params : tuple[Array, Array, float]
        Parameters for the distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    Array
        Computed natural statistic for alpha-divergence.
    """
    psi_theta = p.log_partition(theta, params)

    @partial(jnp.vectorize, signature='(d)->(m)')
    def a_statistic(x):
        inner = inner_fun(x, p, neg_log_likelihood, theta, theta_ast, meas)
        return jnp.exp(inner + 0.5 * (psi_theta - psi_theta_ast)) * p.natural_statistics(x)

    a_theta = A_half_theta(p, neg_log_likelihood, theta, theta_ast, meas, params, psi_theta_ast, z_theta_ast)
    return p.expected_value(a_statistic, theta, params) / a_theta


@filter_jit
def divergence_half_hist(p: NDExponentialFamily,
                         neg_log_likelihood: Callable[[Array, Array], Array],
                         theta_hist: Array,
                         theta_ast: Array,
                         meas: Array,
                         params_hist: tuple,
                         psi_theta_ast: Float,
                         z_theta_ast: Float):
    """
    Computes the half-divergence for a history of parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta_hist : Array
        History of parameter vectors for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params_hist : tuple
        History of parameters for the distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    Array
        History of computed half-divergence values.
    """
    @filter_jit
    def _a_scanned_fun(carry, inputs):
        theta, params = inputs
        d_half = divergence_half(p, neg_log_likelihood, theta, theta_ast, meas, params, psi_theta_ast, z_theta_ast)
        return None, d_half

    _, d_half_history = scan(_a_scanned_fun, None, (theta_hist, params_hist))
    return d_half_history


@filter_jit
def riemannian_norm_half_renyi(theta: Array,
                               args: tuple[NDExponentialFamily, tuple[Array, Array, float],
                                   Callable[[Array, Array], Array], tuple[Array, Array],
                                   Array, Array, float, float]):
    """
    Computes the Riemannian norm for half-Rényi divergence.

    Parameters
    ----------
    theta : Array
        Parameter vector for the distribution.
    args : tuple
        Tuple containing:
        - p : NDExponentialFamily
            Instance of NDExponentialFamily.
        - params : tuple[Array, Array, float]
            Parameters for the distribution.
        - neg_log_likelihood : Callable[[Array, Array], Array]
            Negative log-likelihood function.
        - theta_indices_for_bijection_params : tuple[Array, Array]
            Indices for bijection parameters.
        - theta_ast : Array
            Reference parameter vector.
        - meas : Array
            Measurement data.
        - psi_theta_ast : Float
            Log-partition function evaluated at theta_ast.
        - z_theta_ast : Float
            Log of expected value with respect to theta_ast.

    Returns
    -------
    float
        Computed Riemannian norm.
    """
    (p, params, neg_log_likelihood, theta_indices_for_bijection_params,
     theta_ast, meas, psi_theta_ast, z_theta_ast) = args

    params = update_params(p, theta, params, theta_indices_for_bijection_params)
    fisher = p.fisher_metric(theta, params)
    cartesian_gradient = jacrev(divergence_half, 2)(p, neg_log_likelihood, theta,
                                                    theta_ast, meas, params, psi_theta_ast, z_theta_ast)
    riemannian_gradient = 4 * jnp.linalg.solve(fisher, cartesian_gradient)
    norm_riemannian_gradient = riemannian_norm(riemannian_gradient, fisher)
    return norm_riemannian_gradient


@filter_jit
def riemannian_hessian_of_divergence_half_hist(p: NDExponentialFamily,
                                               neg_log_likelihood: Callable[[Array, Array], Array],
                                               theta_hist: Array,
                                               theta_ast: Array,
                                               meas: Array,
                                               params_hist: tuple,
                                               psi_theta_ast: Float,
                                               z_theta_ast: Float):
    """
    Computes the Riemannian Hessian of half-divergence for a history of parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta_hist : Array
        History of parameter vectors for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params_hist : tuple
        History of parameters for the distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    Array
        History of computed Riemannian Hessian values.
    """
    @filter_jit
    def _a_scanned_fun(carry, inputs):
        theta, params = inputs
        fisher = p.fisher_metric(theta, params)
        inv_fisher = jnp.linalg.inv(fisher)
        cartesian_hessian = hessian(divergence_half, 2)(p, neg_log_likelihood, theta,
                                                        theta_ast, meas, params, psi_theta_ast, z_theta_ast)
        return None, 16 * inv_fisher @ cartesian_hessian @ inv_fisher

    _, hessian_history = scan(_a_scanned_fun, None, (theta_hist, params_hist))
    return hessian_history


@filter_jit
def riemannian_gradient_of_divergence_half_hist(p: NDExponentialFamily,
                                                neg_log_likelihood: Callable[[Array, Array], Array],
                                                theta_hist: Array,
                                                theta_ast: Array,
                                                meas: Array,
                                                params_hist: tuple,
                                                psi_theta_ast: Float,
                                                z_theta_ast: Float):
    """
    Computes the Riemannian gradient of half-divergence for a history of parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta_hist : Array
        History of parameter vectors for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params_hist : tuple
        History of parameters for the distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    tuple
        History of computed Riemannian gradients and their norms.
    """
    @filter_jit
    def _a_scanned_fun(carry, inputs):
        theta, params = inputs
        fisher = p.fisher_metric(theta, params)

        cartesian_gradient = jacrev(divergence_half, 2)(p, neg_log_likelihood, theta,
                                                        theta_ast, meas, params, psi_theta_ast, z_theta_ast)
        riemannian_gradient = 4 * jnp.linalg.solve(fisher, cartesian_gradient)
        norm_riemannian_gradient = riemannian_norm(riemannian_gradient, fisher)
        return None, (riemannian_gradient, norm_riemannian_gradient)

    _, hist = scan(_a_scanned_fun, None, (theta_hist, params_hist))
    riem_grad_history, norm_riem_grad_history = hist
    return riem_grad_history, norm_riem_grad_history


@filter_jit
def kl_divergence_hist(p: NDExponentialFamily,
                       neg_log_likelihood: Callable[[Array, Array], Array],
                       theta_hist: Array,
                       theta_ast: Array,
                       meas: Array,
                       params_hist: tuple,
                       params_ast: tuple,
                       psi_theta_ast: Float,
                       z_theta_ast: Float):
    """
    Computes the Kullback-Leibler (KL) divergence for a history of parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta_hist : Array
        History of parameter vectors for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params_hist : tuple
        History of parameters for the distribution.
    params_ast : tuple
        Parameters for the reference distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    Array
        History of computed KL divergence values.
    """
    @filter_jit
    def _a_scanned_fun(carry, inputs):
        theta, params = inputs
        d_half = kl_divergence(p, neg_log_likelihood, theta, theta_ast, meas, params, params_ast,
                               psi_theta_ast, z_theta_ast)
        return None, d_half

    _, d_half_history = scan(_a_scanned_fun, None, (theta_hist, params_hist))
    return d_half_history


@filter_jit
def convex_conjugate(p: NDExponentialFamily,
                     theta: Array,
                     eta: Array,
                     params: tuple[Array, Array, float]):
    """
    Computes the convex conjugate for an exponential family distribution.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    theta : Array
        Parameter vector for the distribution.
    eta : Array
        Natural parameter vector.
    params : tuple[Array, Array, float]
        Parameters for the distribution.

    Returns
    -------
    float
        Computed convex conjugate value.
    """
    return eta @ theta - p.log_partition(theta, params)


@filter_jit
def convex_conjugate_gradient(p: NDExponentialFamily,
                              theta: Array,
                              eta: Array,
                              params: tuple[Array, Array, float]):
    """
    Computes the gradient of the convex conjugate for an exponential family distribution.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    theta : Array
        Parameter vector for the distribution.
    eta : Array
        Natural parameter vector.
    params : tuple[Array, Array, float]
        Parameters for the distribution.

    Returns
    -------
    Array
        Computed gradient of the convex conjugate.
    """
    fisher = p.fisher_metric(theta, params)
    euclidian_grad = jacrev(convex_conjugate, 1)(p, theta, eta, params)
    return jnp.linalg.solve(fisher, euclidian_grad)


@filter_jit
def convex_conjugate_hist(p: NDExponentialFamily,
                          theta_hist: Array,
                          eta: Array,
                          params_hist: tuple):
    """
    Computes the convex conjugate for a history of parameter vectors.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    theta_hist : Array
        History of parameter vectors for the distribution.
    eta : Array
        Natural parameter vector.
    params_hist : tuple
        History of parameters for the distribution.

    Returns
    -------
    Array
        History of computed convex conjugate values.
    """
    @filter_jit
    def _a_scanned_fun(carry, inputs):
        theta, params = inputs
        cc = convex_conjugate(p, theta, eta, params)
        return None, cc

    _, cc_history = scan(_a_scanned_fun, None, (theta_hist, params_hist))
    return cc_history


@filter_jit
def hessian_inv_grad(p: NDExponentialFamily,
                     neg_log_likelihood: Callable[[Array, Array], Array],
                     theta: Array,
                     theta_ast: Array,
                     meas: Array,
                     params: tuple,
                     psi_theta_ast: Float,
                     z_theta_ast: Float):
    """
    Computes the product of the inverse Riemannian Hessian and the gradient of the half-divergence.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    neg_log_likelihood : Callable[[Array, Array], Array]
        Negative log-likelihood function.
    theta : Array
        Parameter vector for the distribution.
    theta_ast : Array
        Reference parameter vector.
    meas : Array
        Measurement data.
    params : tuple
        Parameters for the distribution.
    psi_theta_ast : Float
        Log-partition function evaluated at theta_ast.
    z_theta_ast : Float
        Log of expected value with respect to theta_ast.

    Returns
    -------
    Array
        Product of the inverse Riemannian Hessian and the gradient of the half-divergence.
    """
    fisher = p.fisher_metric(theta, params)
    euclidian_grad = jacrev(divergence_half, 2)(p, neg_log_likelihood, theta,
                                                theta_ast, meas, params, psi_theta_ast, z_theta_ast)
    euclidean_hess = hessian(divergence_half, 2)(p, neg_log_likelihood, theta,
                                                 theta_ast, meas, params, psi_theta_ast, z_theta_ast)

    return jnp.linalg.solve(0.25 * fisher @ euclidean_hess, euclidian_grad)


@filter_jit
def riemannian_inner(vec_a: Array, vec_b: Array, fisher: Array):
    """
    Computes the Riemannian inner product between two vectors.

    Parameters
    ----------
    vec_a : Array
        First vector.
    vec_b : Array
        Second vector.
    fisher : Array
        Fisher information matrix.

    Returns
    -------
    float
        Riemannian inner product between the two vectors.
    """
    return 4 * vec_a @ fisher @ vec_b


@partial(jnp.vectorize, signature='(d)->()', excluded=[1, ])
@filter_jit
def riemannian_norm(vec: Array, fisher: Array):
    """
    Computes the Riemannian norm of a vector.

    Parameters
    ----------
    vec : Array
        Input vector.
    fisher : Array
        Fisher information matrix.

    Returns
    -------
    float
        Riemannian norm of the input vector.
    """
    return jnp.sqrt(riemannian_inner(vec, vec, fisher))


def get_params(a_params_hist: tuple, an_index: int):
    """
    Extracts parameters from a parameter history at a given index.

    Parameters
    ----------
    a_params_hist : tuple
        History of parameters.
    an_index : int
        Index for extraction.

    Returns
    -------
    tuple
        Extracted parameters at the specified index.
    """
    return a_params_hist[0][an_index], a_params_hist[1][an_index], a_params_hist[2][an_index]


def restructure_em_bayesian_update_output(output_hist: tuple,
                                          n_expectation: int = 20):
    """
    Restructures the output of an EM Bayesian update into separate histories.

    Parameters
    ----------
    output_hist : tuple
        Output history from the EM Bayesian update.
    n_expectation : int, optional
        Number of expectation steps (default is 20).

    Returns
    -------
    tuple
        Restructured histories of theta, parameters, and learning rate.
    """
    theta_hist_bayes = []
    params_0_hist = []
    params_1_hist = []
    params_2_hist = []
    learn_rate_hist_bayes = []
    for i in range(n_expectation):
        theta_hist_bayes.append(output_hist[0][i])
        params_0_hist.append(output_hist[1][0][i])
        params_1_hist.append(output_hist[1][1][i])
        params_2_hist.append(output_hist[1][2][i])
        learn_rate_hist_bayes.append(output_hist[2][i])

    theta_hist_bayes = jnp.vstack(theta_hist_bayes)
    learn_rate_hist_bayes = jnp.hstack(learn_rate_hist_bayes)
    params_0_hist = jnp.vstack(params_0_hist)
    params_1_hist = jnp.concatenate(params_1_hist, axis=0)
    params_2_hist = jnp.hstack(params_2_hist)
    params_hist_bayes = (params_0_hist, params_1_hist, params_2_hist)

    return theta_hist_bayes, params_hist_bayes, learn_rate_hist_bayes


@filter_jit
def update_params(p: NDExponentialFamily,
                  new_theta: Array,
                  old_params: tuple,
                  theta_indices_for_bijection_params: tuple[Array, Array]):
    """
    Updates the parameters of the exponential family distribution.

    Parameters
    ----------
    p : NDExponentialFamily
        Instance of NDExponentialFamily.
    new_theta : Array
        New parameter vector for the distribution.
    old_params : tuple
        Old parameters for the distribution.
    theta_indices_for_bijection_params : tuple
        Indices for bijection parameters.

    Returns
    -------
    tuple
        Updated parameters of the distribution.
    """
    eta = p.extended_statistics_expectation(new_theta, old_params)
    mu = jnp.take(eta, theta_indices_for_bijection_params[0])
    sigma = eta[theta_indices_for_bijection_params[1]] - jnp.outer(mu, mu)
    return mu, sigma, old_params[2]
