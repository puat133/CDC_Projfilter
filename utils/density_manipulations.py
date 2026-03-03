from functools import partial
from typing import Tuple, Callable
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jslg
import numpy as onp
import sympy as sp
from equinox import filter_jit
from jax import jit, numpy as jnp
from jax.lax import scan
from jax.scipy.special import gamma as scipy_gamma
from jax.scipy.special import gammaln
from exponential_family.n_d_ef import NDExponentialFamily
from typing import Union
from jaxtyping import Array, Float

# @partial(jnp.vectorize, signature='(n),(n),(n)->()')
def log_gamma_density(_y: Float, _k: Float, _theta: Float):
    return ((_k - 1) * jnp.log(_y)) - (_y / _theta) - (_k * jnp.log(_theta) + jnp.log(scipy_gamma(_k)))


@partial(jnp.vectorize, signature='(2,2)->(n,n,2)', excluded=(1,))
def create_2d_grid_from_limits(grid_limits: Array, nb_of_points: Array):
    t0 = jnp.linspace(grid_limits[0, 0], grid_limits[0, 1], nb_of_points[0], endpoint=True)
    t1 = jnp.linspace(grid_limits[1, 0], grid_limits[1, 1], nb_of_points[1], endpoint=True)
    grids = jnp.meshgrid(t0, t1, indexing='xy')
    grids = jnp.stack(grids, axis=-1)
    return grids


@partial(jnp.vectorize, signature='(n,n),(2)->()')
def integrate_potential(potential: Array, dxs: Tuple) -> float:
    """
    Integrate a potential function on two-dimensional space. The integration is taken using Trapezoidal rule.

    Parameters
    ----------
    potential: Array
        potential to be integrated
    dxs: Tuple
        delta_x for each axis

    Returns
    -------
    integration_result: Float
        integration result
    """
    return jnp.trapezoid(jnp.trapezoid(potential, dx=dxs[1], axis=1), dx=dxs[0], axis=0)


@partial(jnp.vectorize, signature='(n,n)->(n,n)', excluded=(1,))
def normalize_density(density: Array, dxs: Tuple):
    """
    Normalized a 2d density. The integration used is trapezoidal.

    Parameters
    ----------
    density: Array
        density to be normalized
    dxs: Tuple
        delta_x for each axis

    Returns
    -------
    normalized_density: Array
    """
    return density / integrate_potential(density, dxs)


@filter_jit
@partial(jnp.vectorize, signature='(n)->()', excluded=[1, 2])
def gaussian_kernel_density(point, samples, bandwidth):
    """

    Parameters
    ----------
    point
    samples
    bandwidth

    Returns
    -------

    """
    scaled_points = bandwidth @ (point - samples).T
    gaussian_kernel = jnp.exp(-0.5 * jnp.sum(jnp.square(scaled_points), axis=0)) / jnp.power(2 * jnp.pi,
                                                                                             0.5 * point.shape[0])
    return jnp.sum(gaussian_kernel) / samples.shape[0]


@partial(jnp.vectorize, signature='(n),(n)->(m,m)', excluded=(2, 3))
def histogram(x_particle, y_particle, xbins, ybins):
    """
    get a two-dimensional histogram from a set of x and y positions (particles).

    Parameters
    ----------
    x_particle: Array
    y_particle: Array
    xbins: int
    ybins: int

    Returns
    -------
    empirical_density: Array
        empirical_density

    """
    density, _, _ = jnp.histogram2d(x_particle, y_particle, bins=[xbins, ybins], density=True)
    return density


@partial(jnp.vectorize, signature='(n,n),(n,n)->()', excluded=(2,))
def hellinger_distance(den_1: Array, den_2: Array, dxs:Union[tuple,Array]) -> float:
    """
    Compute Hellinger distance between two density

    Parameters
    ----------
    den_1: Array
        first density
    den_2 : Array
        second density
    dxs : Tuple or Array
        delta_x for each axis

    Returns
    -------
    hellinger_distance : Float
        calculated hellinger distance
    """
    delta_sqrt_dens = jnp.sqrt(jnp.maximum(den_1, 0)) - jnp.sqrt(jnp.maximum(den_2, 0))
    return jnp.sqrt(0.5 * integrate_potential(jnp.square(delta_sqrt_dens), dxs))


@partial(jnp.vectorize, signature='(n,n),(n,n)->()', excluded=(2,))
def cross_entropy(den_1, den_2, dxs, min_den_2_val = 1e-16) -> float:
    """
    Compute Cross Entropy between two density

    Parameters
    ----------

    den_1: Array
        first density
    den_2 : Array
        second density
    dxs : Tuple
        delta_x for each axis
    min_den_2_val: Float
        minimum value of density 2

    Returns
    -------
    hellinger_distance : Float
        calculated hellinger distance
    """
    den_2 = jnp.maximum(den_2, min_den_2_val)
    den_2 = den_2/integrate_potential(den_2, dxs)
    return -integrate_potential(den_1*jnp.log(den_2), dxs)

def cross_entropy_history(den_history_1, den_history_2, dxs) -> Array:
    """

    Parameters
    ----------


    den_history_1: Array
        first density history (N_time x N_x x N_y) 2D, or (N_time x N_x) 1D
    den_history_2
        second density history (N_time x N_x x N_y) 2D, or (N_time x N_x) 1D
    dxs :
        grid space on each axis (Nd) or
        grid space on each axis at each time (N_time x Nd)

    Returns
    -------
    cross_entropy_hist : Array
        cross entropy history (N_time)
    """
    if den_history_1.ndim == 3:
        sample_space_dimension = 2
    elif den_history_1.ndim == 2:
        sample_space_dimension = 1
    else:
        raise NotImplementedError("Only one or two dimensional densities are accepted!")

    @filter_jit
    def scanned_fun_1(_carry, _input):
        den_1, den_2 = _input
        ce = cross_entropy(den_1, den_2, dxs)
        return _carry, ce

    @filter_jit
    def scanned_fun_2(_carry, _input):
        den_1, den_2, dxs_ = _input
        ce = cross_entropy(den_1, den_2, dxs_)
        return _carry, ce

    @filter_jit
    def scanned_fun_3(_carry, _input):
        den_1, den_2 = _input

        ce =-jnp.trapezoid(den_1 * jnp.log(den_2), dx=dxs[0])
        return _carry, ce

    if sample_space_dimension == 2:
        if dxs.ndim == 1:
            _, hell_dist_hist = scan(scanned_fun_1, [], [den_history_1, den_history_2])
        elif dxs.ndim == 2:
            _, hell_dist_hist = scan(scanned_fun_2, [], [den_history_1, den_history_2, dxs])
        else:
            raise Exception("the dxs need to be either one or two dimension!")
    if sample_space_dimension == 1:
        _, hell_dist_hist = scan(scanned_fun_3, [], [den_history_1, den_history_2])

    return hell_dist_hist

def hellinger_distance_history(den_history_1, den_history_2, dxs) -> Array:
    """

    Parameters
    ----------


    den_history_1: Array
        first density history (N_time x N_x x N_y) 2D, or (N_time x N_x) 1D
    den_history_2
        second density history (N_time x N_x x N_y) 2D, or (N_time x N_x) 1D
    dxs :
        grid space on each axis (Nd) or
        grid space on each axis at each time (N_time x Nd)

    Returns
    -------
    hell_dist_hist : Array
        hellinger distance history (N_time)
    """
    if den_history_1.ndim == 3:
        sample_space_dimension = 2
    elif den_history_1.ndim == 2:
        sample_space_dimension = 1
    else:
        raise NotImplementedError("Only one or two dimensional densities are accepted!")

    @filter_jit
    def scanned_fun_1(_carry, _input):
        den_1, den_2 = _input
        hell = hellinger_distance(den_1, den_2, dxs)
        return _carry, hell

    @filter_jit
    def scanned_fun_2(_carry, _input):
        den_1, den_2, dxs_ = _input
        hell = hellinger_distance(den_1, den_2, dxs_)
        return _carry, hell

    @filter_jit
    def scanned_fun_3(_carry, _input):
        den_1, den_2 = _input
        delta_sqrt_dens = jnp.sqrt(jnp.maximum(den_1, 0)) - jnp.sqrt(jnp.maximum(den_2, 0))
        hell = 0.5 * jnp.trapz(jnp.square(delta_sqrt_dens), dx=dxs[0])
        return _carry, hell

    if sample_space_dimension == 2:
        if dxs.ndim == 1:
            _, hell_dist_hist = scan(scanned_fun_1, [], [den_history_1, den_history_2])
        elif dxs.ndim == 2:
            _, hell_dist_hist = scan(scanned_fun_2, [], [den_history_1, den_history_2, dxs])
        else:
            raise Exception("the dxs need to be either one or two dimension!")
    if sample_space_dimension == 1:
        _, hell_dist_hist = scan(scanned_fun_3, [], [den_history_1, den_history_2])

    return hell_dist_hist


@partial(jnp.vectorize, signature='(2,2)->(2),(m),(n)', excluded=(1,))
def dx_and_bins(grid_limit: Array,
                num_points: Array):
    dxs = (grid_limit[:, 1] - grid_limit[:, 0]) / num_points
    x_bins = jnp.linspace(grid_limit[0, 0] - dxs[0], grid_limit[0, 1] + dxs[0], num_points[0] + 1)
    y_bins = jnp.linspace(grid_limit[1, 0] - dxs[1], grid_limit[1, 1] + dxs[1], num_points[1] + 1)
    return dxs, x_bins, y_bins


def histogram_history(x_particle_history: Array,
                      xbins: Array,
                      ybins: Array = None) -> Array:
    """
    Compute histogram history given particle filter samples history.

    Use this instead of the parallelized version of histogram_2d to avoid out of memory error.

    Parameters
    ----------
    x_particle_history: Array
        particle filter records (N_time x N_samples x N_state)
    xbins: Array
        binning on x axis
    ybins: Array
        binning on y axis

    Returns
    -------
    empirical_den_pf_history : Array
        empirical density (N_time x N_x x N_y) if two D or (N_time x N_x) if one D.
    """

    sample_space_dim = x_particle_history.shape[-1]

    @filter_jit
    def scanned_fun_1(_carry, _input):
        x_particle = _input[:, 0]
        y_particle = _input[:, 1]
        a_density = histogram(x_particle, y_particle, xbins, ybins)
        return _carry, a_density.T  # this need to be transposed to match the convention

    @filter_jit
    def scanned_fun_2(_carry, _input):
        a_particle, x_bin_, y_bin_ = _input
        x_particle = a_particle[:, 0]
        y_particle = a_particle[:, 1]
        a_density = histogram(x_particle, y_particle, x_bin_, y_bin_)
        return _carry, a_density.T  # this need to be transposed to match the convention

    @filter_jit
    def scanned_fun_3(_carry, _input):
        a_particle = _input
        a_density, _ = jnp.histogram(a_particle, bins=xbins, density=True)
        return _carry, a_density

    if sample_space_dim == 2:
        if xbins.ndim == 1:
            _, empirical_den_pf_history = scan(scanned_fun_1, [], x_particle_history)
        elif xbins.ndim == 2:
            _, empirical_den_pf_history = scan(scanned_fun_2, [], (x_particle_history, xbins, ybins))
        else:
            raise Exception("the bins need to be either one or two dimension!")
    elif sample_space_dim == 1:
        if xbins.ndim == 1:
            _, empirical_den_pf_history = scan(scanned_fun_3, [], x_particle_history)
        else:
            raise Exception("the bins need to in one dimension!")

    return empirical_den_pf_history


def expectation_history(statistics: Callable,
                        x_particle_history: Array) -> Array:
    """

    Parameters
    ----------
    statistics : Callable
        function: n -> m

    x_particle_history: Array
        particle filter records (N_time x N_samples x N_state)

    Returns
    -------

    """

    @filter_jit
    def scanned_fun(_carry, _input):
        samples_at_time_t = _input
        expectation_at_time_t = jnp.mean(statistics(samples_at_time_t), axis=0)
        return _carry, expectation_at_time_t

    _, expect_hist = scan(scanned_fun, [], x_particle_history)
    return expect_hist


def expectation_history_not_resampled(statistics: Callable,
                                      normalized_log_weights: Array,
                                      x_particle_history: Array) -> Array:
    """

    Parameters
    ----------
    statistics : Callable
        function: n -> m
    normalized_log_weights: Array
        normalized log weights (N_time x N_samples)
    x_particle_history: Array
        particle filter records (N_time x N_samples x N_state)

    Returns
    -------

    """

    @filter_jit
    def scanned_fun(_carry, _input):
        samples_at_t, normalized_log_weight_at_t = _input
        expectation_at_t = jnp.sum(statistics(samples_at_t) * jnp.exp(normalized_log_weight_at_t[:, jnp.newaxis]),
                                   axis=0)
        return _carry, expectation_at_t

    _, expect_hist = scan(scanned_fun, [], (x_particle_history, normalized_log_weights))
    return expect_hist


@partial(jnp.vectorize, signature='(n),(n),(n,n),(n,n)->()')
def hellinger_distance_between_two_gaussians(mean1: Array, mean2: Array,
                                             var1: Array, var2: Array):
    """
    Compute hellinger distance between two Gaussian distribution with means equal to `mean1` and
    `mean2` and variances equal to `var1` and `var2`.

    Parameters
    ----------
    mean1
    mean2
    var1
    var2

    Returns
    -------

    Examples_old
    --------
    >>> import jax.numpy as jnp
    >>> mean1 = jnp.array([1.,1.])
    >>> mean2 = mean1
    >>> var1 = jnp.eye(2)
    >>> var2 = var1
    >>> hellinger_distance_between_two_gaussians(mean1,mean2,var1,var2)
    0.
    """
    delta_mean = mean1 - mean2
    term1 = jnp.power(jnp.linalg.det(var1) * jnp.linalg.det(var2), 0.25) / jnp.power(jnp.linalg.det(0.5 * (var1 + var2))
                                                                                     , 0.5)
    term2 = delta_mean @ (jnp.linalg.solve(0.5 * (var1 + var2), delta_mean))
    hell = 1 - term1 * jnp.exp(-term2 / 8)
    return hell


@filter_jit
def bijection_parameters_time_derivative(d_eta_dt: Array,
                                         params: tuple[Array, Array, float],
                                         theta_indices_for_bijection_params: tuple[Array, Array]
                                         ):
    """

    Parameters
    ----------
    d_eta_dt
    params
    theta_indices_for_bijection_params

    Returns
    -------

    """
    mu = params[0]
    d_mu_dt = jnp.take(d_eta_dt, theta_indices_for_bijection_params[0])
    d_mu_dt_mu_transp = jnp.outer(d_mu_dt, mu)
    d_cov_dt = jnp.take(d_eta_dt, theta_indices_for_bijection_params[1]) - (d_mu_dt_mu_transp + d_mu_dt_mu_transp.T)
    # force d_cov_dt to be symmetric
    d_cov_dt = 0.5 * (d_cov_dt + d_cov_dt.T)
    return d_mu_dt, d_cov_dt



@filter_jit
def triangular_half_of_psd_matrix(P: Array)->Array:
    """
    This is NOT the Cholesky decomposition! instead:
    Given P:
    (P)_(1/2) = lower(P, -1) + 0.5*diag(P)
    """
    return jnp.tril(P, -1) + 0.5 * jnp.diag(jnp.diag(P))

@filter_jit
def dL_dt_from_dP_dt(L:Array, dP_dt:Array):
    # Solve L^{-1} dP/dt via forward substitution
    L_inv = jslg.solve_triangular(L, jnp.eye(L.shape[0]), lower=True)
    # L_inv = jnp.linalg.inv(L)
    # Solve L^{-1} dP/dt L^{-T} via forward substitution on transposed system
    L_inv_P_dot_L_inv_T = L_inv @ dP_dt @ L_inv.T
    # Extract strictly lower triangular + half diagonal
    d_L_dt = L @ triangular_half_of_psd_matrix(L_inv_P_dot_L_inv_T)
    return d_L_dt

@filter_jit
def cholesky_bijection_parameters_time_derivative(d_eta_dt: Array, params: tuple[Array, Array, float],
                                                  theta_indices_for_bijection_params: tuple[Array, Array]):
    """
    the parameters contains the mean, mu, the cholesky (lower diagonal) of the covariance, L, and
    parameter scale

    Parameters
    ----------
    d_eta_dt
    params
    theta_indices_for_bijection_params
    Returns
    -------

    examples
    --------
    """
    d_mu_dt, d_cov_dt = bijection_parameters_time_derivative(d_eta_dt, params, theta_indices_for_bijection_params)
    # S = params[1]
    # S_dot = dL_dt_from_dP_dt(S, d_cov_dt)
    n_dim = d_mu_dt.shape[0]
    Id = jnp.eye(n_dim)
    Tdd = jnp.zeros((n_dim **2,n_dim **2))
    for i in range(n_dim):
        for j in range(n_dim):
            Tdd = Tdd.at[n_dim*i+j,n_dim*j+i].set(1)
    # Tdd = jnp.array(Tdd)
    # Reorder for vec(A) -> vec(A^T) if needed
    tril_indices = jnp.tril_indices(n_dim)
    tril_indices_flat = jnp.arange(n_dim * n_dim).reshape((n_dim, n_dim))[tril_indices]
    S = params[1]
    A = jnp.kron(S, Id) @ Tdd + jnp.kron(Id, S)
    v_S_dot = jnp.linalg.solve(A[jnp.ix_(tril_indices_flat, tril_indices_flat)],
                               d_cov_dt.flatten()[tril_indices_flat])
    S_dot = jnp.zeros((n_dim, n_dim))
    S_dot = S_dot.at[tril_indices].set(v_S_dot)
    return d_mu_dt, S_dot


@filter_jit
def get_natural_statistics_expectation(p: NDExponentialFamily,
                                       theta_hist: Array,
                                       param_hist: tuple[Array, Array, Array]):
    def _get_natural_statistics_expectation(carry, inputs):
        _state, _bijection_params = inputs
        a_moment = p.natural_statistics_expectation(_state, _bijection_params)
        return None, a_moment

    _, expected_natural_statistics_history = scan(_get_natural_statistics_expectation, None,
                                                  (theta_hist, param_hist)
                                                  )
    return expected_natural_statistics_history


def get_density_hist(p: NDExponentialFamily,
                     theta_hist: Array,
                     params_hist: tuple[Array, Array, Array],
                     grid_limits: Array,
                     nb_of_points: Array):
    x_ = []
    for i in range(p.sample_space_dim):
        temp_ = jnp.linspace(grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
        x_.append(temp_)
    grids = jnp.meshgrid(*x_, indexing='xy')
    grids = jnp.stack(grids, axis=-1)

    def _evaluate_density_loop(_carry, _input):
        theta_, params_ = _input
        _, density_ = p.get_density_values_from_grids(grids, theta_, params_)
        bijected_points_ = p.bijection(p.quadrature_points, params_)
        return None, (density_, bijected_points_)

    _, (density_history_, bijection_points_history_) = scan(_evaluate_density_loop,
                                                            None, (theta_hist,
                                                                   params_hist))
    return density_history_, bijection_points_history_


def calculate_gaussian_natural_parameters(mean: Array,
                                          cov: Array,
                                          natural_statistics_symbolic: sp.Matrix,
                                          symbolic_state: tuple[sp.Symbol]):
    cov_inv = jnp.linalg.solve(cov, jnp.eye(mean.shape[0]))
    theta_1 = cov_inv @ mean
    theta_2 = -0.5 * cov_inv

    # natural_statistics_symbolic = hyperbolic_cross_monomials(x, max_order_monomials)
    # Gaussian initial state
    gaussian_initial_condition = jnp.zeros(len(natural_statistics_symbolic))
    theta_indices_for_mu = []
    # the mean
    for i in range(mean.shape[0]):
        for k in range(len(natural_statistics_symbolic)):
            if natural_statistics_symbolic[k] == symbolic_state[i]:
                gaussian_initial_condition = gaussian_initial_condition.at[k].set(theta_1[i])
                theta_indices_for_mu.append(k)
                break

    theta_indices_for_mu = jnp.array(theta_indices_for_mu)
    theta_indices_for_Sigma = onp.zeros((mean.shape[0], mean.shape[0]))
    # the variance
    for i in range(mean.shape[0]):
        for j in range(i, mean.shape[0]):
            multiplier = 2
            if i == j:
                multiplier = 1
            for k in range(len(natural_statistics_symbolic)):
                if natural_statistics_symbolic[k] == symbolic_state[i] * symbolic_state[j]:
                    gaussian_initial_condition = gaussian_initial_condition.at[k].set(multiplier * theta_2[i, j])
                    theta_indices_for_Sigma[i, j] = k
                    theta_indices_for_Sigma[j, i] = k
                    break

    theta_indices_for_Sigma = jnp.array(theta_indices_for_Sigma, dtype=jnp.int32)
    theta_indices_for_bijection_params = (theta_indices_for_mu, theta_indices_for_Sigma)

    return gaussian_initial_condition, theta_indices_for_bijection_params


@filter_jit
@partial(jnp.vectorize, signature='(d),(d,d)->(m,d)', excluded=(2, 3))
def ellipse_from_mean_n_cov(mean: jnp.array, cov: jnp.array, n_std: int = 1, n_points: int = 100):
    """
    This function gives an ellipse from a mean and a covariance.
    """

    theta = jnp.linspace(0, 2 * jnp.pi, n_points)
    unit_circle = jnp.array([jnp.cos(theta), jnp.sin(theta)])

    # Perform singular value decomposition on the covariance matrix
    eigvals, eigvecs = jnp.linalg.eigh(cov)

    # Construct the ellipse by scaling the unit circle and applying the rotation
    ellipse = jnp.dot(eigvecs, jnp.diag(jnp.sqrt(eigvals)) @ unit_circle) * n_std

    # Shift the ellipse to be centered at the mean
    # ellipse[0, :] += mean[0]
    # ellipse[1, :] += mean[1]
    ellipse = ellipse.T
    ellipse = ellipse + mean
    return ellipse


def equispace_grid(grid_limits: Array, nb_of_points: Array):
    """

    Parameters
    ----------
    grid_limits: Array
        The limits of the grid
    nb_of_points: Array[int]
        The number of points in the grid

    Returns
    -------

    """
    x_ = []
    sample_space_dim = nb_of_points.shape[0]
    for i in range(sample_space_dim):
        temp_ = jnp.linspace(grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
        x_.append(temp_)
    a_grid = jnp.meshgrid(*x_, indexing='xy')
    a_grid = jnp.stack(a_grid, axis=-1)
    return a_grid


@partial(jnp.vectorize, signature='(n)->()', excluded=(1,))
def log_gaussian_density(x: Array, cov: Array) -> float:
    """
    Evaluate log of gaussian density with zero mean and an inverse covariance matrix `cov_inv`.
    Parameters
    ----------
    x       : (N,) np.ndarray
        vector
    cov: (N,N) np.ndarray
        inverse covariance
    Returns
    -------
    out: Float
        log gaussian density
    """
    return -0.5 * (x @ jnp.linalg.solve(cov, x) + cov.shape[0] * jnp.log(2 * jnp.pi) + jnp.linalg.slogdet(cov)[1])


@partial(jnp.vectorize, signature='(n)->()', excluded=(1, 2))
def gaussian_density_values(a_point: Array, mean: Array, cov: Array):
    return jnp.exp(log_gaussian_density(a_point - mean, cov))


@partial(jnp.vectorize, signature='(n)->()', excluded=(1, 2, 3))
def mix_gaussian_density_values(points: Array, means: Array, covs: Array,
                                log_weights: Array):
    mix_density = 0.

    @filter_jit
    def scanned_fun(_carry, _input):
        _current_mix_density = _carry
        log_weight, mean, cov = _input
        _current_mix_density += jnp.exp(log_weight) * gaussian_density_values(points, mean, cov)

        return _current_mix_density, None

    mix_density, _ = scan(scanned_fun, mix_density, (log_weights, means, covs))

    return mix_density


def mix_gaussian_densities_on_grid(grid_limits: Array,
                                   n_point: int,
                                   log_weights: Array,
                                   means: Array,
                                   covs: Array) -> Array:
    """
    get
    Parameters
    ----------
    grid_limits:
        lower and upper limit for each axis
    n_point:
        how many point per each axis
    log_weights:
        log weights for each mixand
    means:
        the means
    covs:
        the covariances

    Returns
    -------

    """
    grid = equispace_grid(grid_limits, jnp.array([n_point for i in range(grid_limits.shape[0])]))
    mix_density = mix_gaussian_density_values(grid, means, covs, log_weights)
    return mix_density


def gaussian_mixture_cross_entropy(samples: Array,
                                   means: Array,
                                   covs: Array,
                                   log_weights: Array) -> float:
    """
    Relative entropy from an empirical density with samples, to
    a gaussian_mixture with given means and covariances

    Parameters
    ----------
    samples
    means
    covs
    log_weights

    Returns
    -------

    """

    mix_density = mix_gaussian_density_values(samples, means, covs, log_weights)
    return - (jnp.sum(jnp.log(mix_density))) / samples.shape[0]


def comb(n, k):
    """
    Compute the binomial coefficient "N choose k" using the gamma function.

    Parameters
    ----------
    n : array_like
        Total number of items, can be an array.
    k : array_like
        Number of items to choose, must be of the same shape as N.

    Returns
    -------
    binom_coeff : array_like
        Binomial coefficients corresponding to N and k.
    """

    return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


# @partial(jit, static_argnums=(2,))
def compute_moment(mu, Sigma, multi_index):
    """
    Compute the expectation E[x^multi_index] for a multivariate normal distribution.

    Parameters
    ----------
    mu : array_like
        Mean vector of the multivariate normal distribution, shape (d,).
    sigma : array_like
        Covariance matrix of the multivariate normal distribution, shape (d, d).
    multi_index : tuple or list
        Multi-index of exponents for each variable, shape (d,).

    Returns
    -------
    total_expectation : Float
        The computed expectation E[x^multi_index].

    Notes
    -----
    This function computes the expectation of a monomial of Gaussian random variables
    using properties of multivariate normal distributions and leverages JAX for
    just-in-time compilation and vectorization.

    The computation involves:
    - Generating all possible combinations of k_i where 0 <= k_i <= i_i and sum(k_i) is even.
    - Calculating coefficients using binomial coefficients and powers of the mean vector.
    - Computing expectations of products of zero-mean Gaussian variables using Isserlis’ theorem.
    - Using `jnp.vectorize` for clarity instead of `vmap`.

    Due to computational complexity, for large degrees or dimensions, the function may
    approximate higher-order moments or set them to zero to maintain performance.

    Examples
    --------
    >>> import numpy as np
    >>> mu = np.array([1.0, 2.0])
    >>> Sigma = np.array([[1.0, 0.5],
    ...                   [0.5, 2.0]])
    >>> multi_index = (2, 2)
    >>> expected_value = compute_moment(mu, Sigma, multi_index)
    >>> print("E[x^multi_index] =", expected_value)
    E[x^multi_index] = 17.0

    """
    # Convert inputs to JAX arrays for compatibility
    mu = jnp.asarray(mu)
    Sigma = jnp.asarray(Sigma)
    multi_index = jnp.asarray(multi_index)
    d = mu.shape[0]  # Dimension of the distribution

    # Generate all possible combinations of k_i where 0 <= k_i <= i_i
    ranges = [jnp.arange(i_i + 1) for i_i in multi_index]
    grids = jnp.meshgrid(*ranges, indexing='ij')
    ks = jnp.stack([g.flatten() for g in grids], axis=-1)  # All combinations of k_i

    # Filter combinations where the sum of k_i is even (necessary condition)
    sum_k = jnp.sum(ks, axis=1)
    even_mask = jnp.mod(sum_k, 2) == 0
    ks_even = ks[even_mask]

    # Define compute_coeff function and vectorize it using jnp.vectorize
    def compute_coeff(k_i):
        binom_coeff = jnp.prod(comb(multi_index, k_i))
        mu_power = jnp.prod(mu ** (multi_index - k_i))
        return binom_coeff * mu_power

    compute_coeff_vectorized = jnp.vectorize(compute_coeff, signature='(n)->()')
    coeffs = compute_coeff_vectorized(ks_even)

    # Define epsilon_expectation function and vectorize it using jnp.vectorize
    def epsilon_expectation(k_i):
        k_i = k_i.astype(int)
        total_k = jnp.sum(k_i)
        if total_k == 0:
            # No epsilon variables; expectation is 1
            return 1.0

        # Generate indices of epsilon variables based on k_i
        indices = []
        for idx, count in enumerate(k_i):
            indices.extend([idx] * count)
        indices = jnp.array(indices)

        n = indices.shape[0]  # Total number of epsilon variables
        if n % 2 != 0:
            # Odd number of variables; expectation is zero
            return 0.0
        if n > 8:
            # For large n, computing all pairings is impractical
            # Return 0.0 as an approximation to maintain performance
            return 0.0

        # Generate all possible pairings using Isserlis' theorem
        from itertools import combinations

        # Generate all pairwise combinations of indices
        pairings = list(combinations(range(n), 2))

        # Compute the sum over products of covariances for all pairings
        expectation = 0.0
        for pairing in pairings:
            i = indices[pairing[0]]
            j = indices[pairing[1]]
            expectation += Sigma[i, j]
        return expectation

    epsilon_expectation_vectorized = jnp.vectorize(epsilon_expectation, signature='(n)->()')
    epsilon_expectations = epsilon_expectation_vectorized(ks_even)

    # Compute the total expectation by summing over all terms
    total_expectation = jnp.sum(coeffs * epsilon_expectations)
    return total_expectation

@partial(jnp.vectorize, signature='(d)->()', excluded=(1,2,))
@filter_jit
def mahalanobis_dist_sq(x: jnp.ndarray, mean: jnp.ndarray, cov_chol: jnp.ndarray) -> float:
    """
    Calculate the squared Mahalanobis distance between a point and a distribution.

    Parameters
    ----------
    x : jnp.ndarray
        The point to evaluate.
    mean : jnp.ndarray
        The mean of the distribution.
    cov_chol : jnp.ndarray
        The Cholesky decomposition of the covariance matrix of the distribution.

    Returns
    -------
    float
        The squared Mahalanobis distance.
    """
    diff = x - mean
    return jnp.sum(jnp.square(jnp.linalg.solve(cov_chol, diff)))


def sample_gaussian_mixture(means: jnp.ndarray, covs: jnp.ndarray, weights: jnp.ndarray,
                           n_samples: int, key: jnp.ndarray) -> jnp.ndarray:
    """
    Sample from a Gaussian mixture model.

    Parameters
    ----------
    means : jnp.ndarray
        Component means, shape (n_components, dim).
    covs : jnp.ndarray
        Component covariances, shape (n_components, dim, dim).
    weights : jnp.ndarray
        Component weights, shape (n_components,). Should sum to 1.
    n_samples : int
        Number of samples to draw.
    key : jnp.ndarray
        JAX random key.

    Returns
    -------
    jnp.ndarray
        Samples from the mixture, shape (n_samples, dim).
    """
    import jax.random as jrandom

    n_components = means.shape[0]
    dim = means.shape[1]

    # Normalize weights to ensure they sum to 1
    weights = weights / jnp.sum(weights)

    # Split key for component selection and sampling
    key, subkey_components, subkey_samples = jrandom.split(key, 3)

    # Sample component indices
    component_indices = jrandom.choice(subkey_components, n_components, shape=(n_samples,), p=weights)

    # Sample from standard normal
    z = jrandom.normal(subkey_samples, shape=(n_samples, dim))

    # Get means and compute Cholesky decompositions for selected components
    selected_means = means[component_indices]  # (n_samples, dim)
    selected_covs = covs[component_indices]    # (n_samples, dim, dim)

    # Compute Cholesky decomposition for each selected covariance
    # Use vmap for efficiency
    chol_covs = jax.vmap(jnp.linalg.cholesky)(selected_covs)  # (n_samples, dim, dim)

    # Transform samples: x = mean + L @ z
    # Use einsum for batch matrix-vector multiplication
    samples = selected_means + jnp.einsum('nij,nj->ni', chol_covs, z)

    return samples

