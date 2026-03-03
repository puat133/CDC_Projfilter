from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
from equinox import filter_jit

from exponential_family.ef import ExponentialFamily
from jaxtyping import Array

class NDExponentialFamily(ExponentialFamily, ABC):
    """ N-dimensional Exponential Family Distribution

    A class to represent a n-dimensional exponential family distribution, with integration
    performed via Quasi Monte Carlo or Sparse-Grid quadrature methods.

    Attributes
    ----------
    _quadrature_points : Array
        Points used for numerical quadrature
    _bijected_points : Array
        Transformed quadrature points after applying bijection
    _bijected_log_dvolume : Array
        Log of volume element after bijection transformation
    _s_rule : str
        String identifier for the quadrature rule used
    _direct_calculation : bool
        Whether to use direct calculation instead of automatic differentiation
    _natural_statistic_expectation : Callable
        Function to compute expectation of natural statistics
    _extended_statistic_expectation : Callable
        Function to compute expectation of extended statistics
    _fisher_metric : Callable
        Function to compute Fisher information metric

    Methods
    -------
    natural_statistics_expectation
        Computes expectation of natural statistics
    extended_statistics_expectation
        Computes expectation of extended statistics
    fisher_metric
        Computes Fisher information metric
    integrate_partition
        Integrates partition function
    integrate_partition_extended
        Integrates extended partition function
    get_density_values
        Evaluates density on grid points
    integrate
        General purpose integration method
    integrate_exponential_fun
        Integrates exponential of given function
    """

    _quadrature_points: Array
    _bijected_points: Array
    _bijected_log_dvolume: Array
    _s_rule: str
    _direct_calculation: bool

    _natural_statistic_expectation: Callable[[Array, tuple], Array]
    _extended_statistic_expectation: Callable[[Array, tuple], Array]
    _fisher_metric: Callable[[Array, tuple], Array]


    def __init__(self,
                 sample_space_dimension: int,
                 bijection: Callable[[Array, tuple], Array],
                 statistics: Callable[[Array], Array],
                 remaining_statistics: [[Array], Array] = None,
                 bijection_parameters: tuple = None,
                 theta_indices_for_bijection_params: tuple = (jnp.array([0, ], dtype=jnp.int32),
                                                              jnp.array([1, ], dtype=jnp.int32)),
                 direct_calculation: bool = False
                 ):

        super().__init__(sample_space_dimension=sample_space_dimension,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params)

        # default value to be realized in the implemented class
        self._quadrature_points = jnp.empty((1,))
        self._bijected_points = jnp.empty((1,))
        self._bijected_log_dvolume = jnp.empty((1,))
        self._s_rule = ""
        self._direct_calculation = direct_calculation

        # default case

        if self._direct_calculation:
            self._natural_statistic_expectation = direct_natural_statistics_expectation
            self._extended_statistic_expectation = direct_extended_statistics_expectation
            self._fisher_metric = direct_fisher_metric
        else:
            self._natural_statistic_expectation = natural_statistics_expectation
            self._extended_statistic_expectation = extended_statistics_expectation
            self._fisher_metric = fisher_metric

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
    @filter_jit
    def natural_statistics_expectation(self, theta: Array, bijection_params: tuple):
        return self._natural_statistic_expectation(self, theta, bijection_params)

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
    @filter_jit
    def extended_statistics_expectation(self, theta: Array, bijection_params: tuple):
        return self._extended_statistic_expectation(self, theta, bijection_params)

    @partial(jnp.vectorize, signature='(m)->(m,m)', excluded=[0, 2])
    @filter_jit
    def fisher_metric(self, theta: Array, bijection_params: tuple):
        return self._fisher_metric(self, theta, bijection_params)

    @property
    def direct_calculation(self):
        return self._direct_calculation

    @property
    @abstractmethod
    def srule(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def bijected_points(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def quadrature_points(self):
        raise NotImplementedError

    @filter_jit
    def inner(self, theta: Array, bijection_params: tuple) -> Array:
        x_tilde = self._quadrature_points
        x = self._bijection(x_tilde, bijection_params)
        inner_val = self.log_dvolume(x_tilde, bijection_params) + \
                    self._natural_statistics(x) @ theta
        return inner_val

    @filter_jit
    def extended_inner(self, theta_extended: Array, bijection_params: tuple) -> Array:
        x_tilde = self._quadrature_points
        x = self._bijection(x_tilde, bijection_params)
        inner_val = self.log_dvolume(x_tilde, bijection_params) + \
                    self._extended_statistics(x) @ theta_extended
        return inner_val



    def get_density_values(self, grid_limits: Array, theta: Array, nb_of_points: Array
                           , bijection_params: tuple) -> \
            tuple[Array, Array]:
        """
        Evaluates the probability density function on a grid of points.

        Parameters
        ----------
        grid_limits : Array
            Array of shape (d, 2) containing lower and upper bounds for each dimension
        theta : Array
            Natural parameters of the exponential family distribution
        nb_of_points : Array
            Number of grid points to use in each dimension
        bijection_params : tuple
            Parameters for the bijection transformation

        Returns
        -------
        tuple[Array, Array]
            A tuple containing:
            - grids: Meshgrid of evaluation points
            - density: Evaluated probability density at each grid point
        """
        x_ = []
        for i in range(self._sample_space_dim):
            temp_ = jnp.linspace(grid_limits[i, 0], grid_limits[i, 1], nb_of_points[i], endpoint=True)
            x_.append(temp_)
        grids = jnp.meshgrid(*x_, indexing='xy')
        grids = jnp.stack(grids, axis=-1)

        return self.get_density_values_from_grids(grids, theta, bijection_params)

    def get_density_values_from_grids(self, grids: Array, theta: Array, bijection_params: tuple):
        """
                Evaluates the probability density function on a pre-defined grid of points.

                Parameters
                ----------
                grids : Array
                    Pre-defined grid points at which to evaluate the density
                theta : Array
                    Natural parameters of the exponential family distribution
                bijection_params : tuple
                    Parameters for the bijection transformation

                Returns
                -------
                tuple[Array, Array]
                    A tuple containing:
                    - grids: The input grid points
                    - density: Evaluated probability density at each grid point
                """
        c_ = self.natural_statistics(grids)

        @filter_jit
        def _evalulate_density(theta_):
            psi_ = self.log_partition(theta_, bijection_params)
            density_ = jnp.exp(c_ @ theta_ - psi_)
            return density_

        density = _evalulate_density(theta)
        return grids, density

    @abstractmethod
    def compute_D_part_for_fisher_metric(self, c: Array, exp_c_theta: Array, dvol: Array):
        """
        Compute the D part of the Fisher metric calculation.

        This method is part of the direct calculation of the Fisher metric without using
        automatic differentiation. It computes a component needed for the Fisher information matrix.

        Parameters
        ----------
        c : Array
            Natural statistics evaluated at the bijected quadrature points
        exp_c_theta : Array
            Exponential of the inner product of natural statistics and parameters
        dvol : Array
            Log volume element after bijection transformation

        Returns
        -------
        Array
            The D matrix component of the Fisher metric calculation
        """
        raise NotImplementedError

    @filter_jit
    def integrate(self, fun: Callable[[Array, ], Array], bijection_params: tuple) -> Array:
        """
        Perform numerical integration of a given function over the sample space.

        This method transforms the integration domain using the specified bijection and applies
        the appropriate volume element corrections before performing numerical integration.

        Parameters
        ----------
        fun : Callable[[Array, ], Array]
            The function to be integrated. Should take an array of points and return an array
            of function values.
        bijection_params : tuple
            Parameters for the bijection transformation that maps the integration domain.

        Returns
        -------
        Array
            The numerical result of the integration.
        """
        x_tilde = self._quadrature_points
        integrand = jnp.exp(self.log_dvolume(x_tilde, bijection_params)) * fun(
            self.bijection(x_tilde, bijection_params))
        return self.numerical_integration(integrand)

    @filter_jit
    def integrate_exponential_fun(self, log_fun: Callable[[Array, ], Array], bijection_params: tuple) \
            -> tuple[Array, Array]:
        """
        Performs numerical integration of the exponential of a log function.

        This method computes integrals of the form ∫exp(log_fun(x))dx using numerical quadrature,
        with appropriate transformations and normalizations for numerical stability.

        Parameters
        ----------
        log_fun : Callable[[Array, ], Array]
            The log of the function to be integrated. Should take an array of points and return
            an array of log values.
        bijection_params : tuple
            Parameters for the bijection transformation that maps the integration domain.

        Returns
        -------
        tuple[Array, Array]
            A tuple containing:
            - res: The normalized integral result
            - max_inner: Maximum value of the inner product used for numerical stability
        """
        x_tilde = self._quadrature_points
        bijected_x_points = self._bijection(x_tilde, bijection_params)
        inner = self.log_dvolume(x_tilde, bijection_params) + log_fun(bijected_x_points)
        max_inner = jnp.max(inner)
        normalized_inner = jnp.exp(inner - max_inner)

        res = self.numerical_integration(normalized_inner)
        return res, max_inner


@partial(jnp.vectorize, signature='(m)->(m,m)', excluded=[0, 2])
@filter_jit
def fisher_metric(p: NDExponentialFamily, theta: Array, bijection_param: tuple) \
        -> Array:
    """
    Computes the Fisher information metric for the exponential family distribution.

    This method uses automatic differentiation through JAX to compute the Hessian of the log partition
    function, which gives the Fisher information metric. The Fisher metric is a Riemannian metric on
    the statistical manifold that measures the informational difference between nearby distributions.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution object
    theta : Array
        Natural parameters of the exponential family
    bijection_param : tuple
        Parameters for the bijection transformation

    Returns
    -------
    Array
        The Fisher information metric evaluated at the given parameters, returned as a matrix
    """
    return p.log_partition_hess(theta, bijection_param)


@partial(jnp.vectorize, signature='(m)->(m,m)', excluded=[0, 2])
@filter_jit
def direct_fisher_metric(p: NDExponentialFamily, theta: Array, bijection_param: tuple) \
        -> Array:
    """
    Compute Fisher information metric without using automatic differentiation.

    Instead of using JAX's automatic differentiation, this method directly computes the Fisher
    information metric using explicit formulas. The result is mathematically equivalent to the
    autodiff version but may be more computationally efficient in some cases.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution object
    theta : Array
        Natural parameters of the exponential family
    bijection_param : tuple
        Parameters for the bijection transformation

    Returns
    -------
    Array
        The Fisher information metric evaluated at the given parameters, returned as a matrix.
        This is a symmetric positive definite matrix representing the local curvature of the
        statistical manifold at the given parameters.
    """
    x_tilde = p.quadrature_points
    c = p.natural_statistics(p.bijection(x_tilde, bijection_param))
    inner = p.inner(theta, bijection_param)
    max_inner = jnp.max(inner)  # exponential max_inner will cancels out
    exp_c_theta = jnp.reshape(jnp.exp(inner - max_inner), (p.quadrature_points.shape[0], 1))
    dvol = p.log_dvolume(x_tilde, bijection_param)[:, jnp.newaxis]
    expectation_of_exp_c_theta_dv = p.numerical_integration((exp_c_theta * dvol).T)
    expectation_of_exp_c_theta_c_dv = p.numerical_integration((exp_c_theta * c * dvol).T, axis=0)
    D = p.compute_D_part_for_fisher_metric(c, exp_c_theta, dvol)
    return (1 / expectation_of_exp_c_theta_dv) * ((-1 / expectation_of_exp_c_theta_dv) *
                                                  jnp.outer(expectation_of_exp_c_theta_c_dv,
                                                            expectation_of_exp_c_theta_c_dv) + D)


@partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
@filter_jit
def natural_statistics_expectation(p: NDExponentialFamily, theta: Array, bijection_param: tuple) \
        -> Array:
    """
    Compute the expectation of natural statistics using automatic differentiation.

    This method computes the expectation of the natural statistics with respect to the exponential
    family distribution by taking the Jacobian of the log partition function. This is a fundamental
    property of exponential families where the expectation of the sufficient statistics equals the
    derivatives of the log partition function.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution object
    theta : Array
        Natural parameters of the exponential family
    bijection_param : tuple
        Parameters for the bijection transformation

    Returns
    -------
    Array
        The expected value of the natural statistics under the exponential family distribution
        with the given parameters
    """
    return p.log_partition_jac(theta, bijection_param)


@partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
@filter_jit
def direct_natural_statistics_expectation(p: NDExponentialFamily, theta: Array, bijection_param: tuple) \
        -> Array:
    """
    Compute the expectation of natural statistics without using automatic differentiation.

    This method provides a direct computation of the expectation of natural statistics for the
    exponential family distribution, avoiding the use of automatic differentiation. This can be
    computationally more efficient in some cases while producing identical results to the
    autodiff version.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution object
    theta : Array
        Natural parameters of the exponential family
    bijection_param : tuple
        Parameters for the bijection transformation

    Returns
    -------
    Array
        The expected value of the natural statistics under the exponential family distribution
        with the given parameters, computed using direct calculation methods
    """
    x_tilde = p.quadrature_points
    c = p.natural_statistics(p.bijection(x_tilde, bijection_param))
    inner = p.inner(theta, bijection_param)
    max_inner = jnp.max(inner) # exponential max_inner will cancels out
    exp_c_theta = jnp.reshape(jnp.exp(inner - max_inner), (p.quadrature_points.shape[0], 1))
    dvol = p.log_dvolume(x_tilde, bijection_param)[:, jnp.newaxis]
    expectation_of_exp_c_theta_dv = p.numerical_integration((exp_c_theta * dvol).T)
    expectation_of_exp_c_theta_c_dv = p.numerical_integration((exp_c_theta * c * dvol).T, axis=0)
    return expectation_of_exp_c_theta_c_dv / expectation_of_exp_c_theta_dv


@partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
@filter_jit
def extended_statistics_expectation(p: NDExponentialFamily, theta: Array, bijection_param: tuple) \
        -> Array:
    """
    Compute the expectation of extended statistics using automatic differentiation.

    This method computes the expectation of both natural and remaining statistics using the
    Jacobian of the extended log partition function. The extended statistics include additional
    moments beyond the natural statistics of the exponential family.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution object
    theta : Array
        Natural parameters of the exponential family
    bijection_param : tuple
        Parameters for the bijection transformation

    Returns
    -------
    Array
        The expected value of the extended statistics under the exponential family distribution
        with the given parameters, computed using automatic differentiation
    """
    return p.log_partition_extended_jac(jnp.pad(theta, (0, p.remaining_moments_num)), bijection_param)


@partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
@filter_jit
def direct_extended_statistics_expectation(p: NDExponentialFamily, theta_extended: Array, bijection_param: tuple) \
        -> Array:
    """
    Compute the expectation of extended statistics without using automatic differentiation.

    This method provides a direct computation of extended statistics expectations, which include
    both natural and remaining statistics, without relying on automatic differentiation. The
    computation is performed using explicit numerical integration and produces identical results
    to the autodiff version.

    Parameters
    ----------
    p : NDExponentialFamily
        The exponential family distribution object
    theta_extended : Array
        Extended natural parameters including both natural and remaining parameters
    bijection_param : tuple
        Parameters for the bijection transformation

    Returns
    -------
    Array
        The expected value of the extended statistics under the exponential family distribution
        with the given parameters, computed using direct calculation methods
    """
    x_tilde = p.quadrature_points
    c = p.extended_statistics(p.bijection(x_tilde, bijection_param))
    inner = p.extended_inner(theta_extended, bijection_param)
    max_inner = jnp.max(inner)  # exponential max_inner will cancels out
    exp_c_theta = jnp.reshape(jnp.exp(inner - max_inner), (p.quadrature_points.shape[0], 1))
    dvol = p.log_dvolume(x_tilde, bijection_param)[:, jnp.newaxis]
    expectation_of_exp_c_theta_dv = p.numerical_integration((exp_c_theta * dvol).T)
    expectation_of_exp_c_theta_c_dv = p.numerical_integration((exp_c_theta * c * dvol).T, axis=0)
    return expectation_of_exp_c_theta_c_dv / expectation_of_exp_c_theta_dv
