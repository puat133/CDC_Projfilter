from collections.abc import Callable

from exponential_family.n_d_ef import NDExponentialFamily
import jax.numpy as jnp
from abc import ABC, abstractmethod
from jax.scipy.special import erfinv
from utils.vectorized import inner
from equinox import filter_jit
from jaxtyping import Array

class NDExponentialFamilySparse(NDExponentialFamily, ABC):
    """
    A sparse grid exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m`
        where integration of log partition function is solved via a sparse grid quadrature.

        This class implements a sparse grid quadrature approach to computing integrals for the exponential family.
        The sparse grid reduces the curse of dimensionality compared to full tensor product quadrature, while
        maintaining good accuracy for smooth integrands.
    Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
    of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).

    Parameters
    ----------
    sample_space_dimension : int
        sample space dimension
    sparse_grid_level : int
        sparse grid level
    statistics : Callable[[Array], Array]
        statistics function
    remaining_statistics: Callable[[Array], Array]
        remaining statistics function
    weight_cut_off: float
        weight cut off for the smolyak sparse grid
    bijection: Callable[[Array, tuple], Array]
        bijection function
    bijection_parameters: tuple
        bijection parameters
    epsilon : float
        epsilon
    s_rule   : str
        sparse integration rule
    weight_cut_off: float
        weight cut off for the smolyak sparse grid
    """

    _spg_level: int
    _epsilon: float
    _weight_cut_off: float
    _nodes_num: int
    _quadrature_weights: Array

    def __init__(self,
                 sample_space_dimension: int,
                 sparse_grid_level: int,
                 bijection: Callable[[Array, tuple], Array],
                 statistics: Callable[[Array], Array],
                 remaining_statistics: [[Array], Array] = None,
                 bijection_parameters: tuple = None,
                 epsilon: float = 0,
                 s_rule: str = "",
                 weight_cut_off: float = None,
                 theta_indices_for_bijection_params: tuple = (jnp.array([0, ], dtype=jnp.int32),
                                                              jnp.array([1, ], dtype=jnp.int32)),
                 direct_calculation: bool = False
                 ):


        super().__init__(sample_space_dimension=sample_space_dimension,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params,
                         direct_calculation=direct_calculation)
        self._spg_level = sparse_grid_level
        self._s_rule = s_rule
        self._epsilon = epsilon
        self._weight_cut_off = weight_cut_off
        points, weights = self.initialize_sparse_grid()

        # To avoid nan in bijection result
        # Gauss-Hermite Quadrature is not confined in [-1.1]
        if self._s_rule.lower() != "gauss-hermite":
            mask = jnp.linalg.norm(points, ord=jnp.inf, axis=-1) < 1 - self._epsilon
            # we directly scale the points from [-1, 1] to erfinv([-1,1])
            self._quadrature_points = erfinv(jnp.asarray(points[mask]))
            # hence we need to multiply the weights by the Jacobian of the transformation which is just the gaussian density.
            self._quadrature_weights = jnp.asarray(weights[mask]) * jnp.prod(0.5 * jnp.sqrt(jnp.pi) *
                                                                             jnp.exp(
                                                                                 jnp.square(self._quadrature_points)),
                                                                             axis=1)
        else:
            self._quadrature_points = jnp.asarray(points)
            self._quadrature_weights = jnp.asarray(weights) * jnp.exp(inner(self._quadrature_points,
                                                                            self._quadrature_points))

        self._nodes_num = self._quadrature_weights.shape[0]
        self._bijected_points = self._bijection(self._quadrature_points, self.bijection_params)
        self._bijected_log_dvolume = jnp.reshape(self.log_dvolume(self._quadrature_points, self.bijection_params),
                                                 (self._quadrature_points.shape[0], 1))

    @abstractmethod
    def initialize_sparse_grid(self):
        return [], []

    @property
    def srule(self):
        return self._s_rule

    @property
    def quadrature_weights(self):
        return self._quadrature_weights

    @property
    def nodes_number(self):
        return self._nodes_num

    @property
    def bijected_points(self):
        return self._bijected_points

    @property
    def quadrature_points(self):
        return self._quadrature_points

    @property
    def sparse_grid_level(self):
        return self._spg_level

    @filter_jit
    def numerical_integration(self, numerical_values: Array, axis=None):
        """
        Performs numerical integration on the given numerical values using quadrature weights.

        Parameters
        ----------
        numerical_values : Array
            Values to be integrated
        axis : int, optional
            Axis along which to compute mean, by default None

        Returns
        -------
        Array
            The result of numerical integration. If axis is None, returns values multiplied by weights.
            If axis is specified, returns mean along that axis.
        """
        if not axis:
            result = numerical_values @ self._quadrature_weights
        else:
            result = jnp.mean(numerical_values @ self._quadrature_weights, axis=axis)
        return result

    @filter_jit
    def compute_D_part_for_fisher_metric(self, c: Array, exp_c_theta: Array, dvol: Array):
        """
        Computes the D part of the Fisher metric.

        Parameters
        ----------
        c : Array
            Sufficient statistics evaluated at quadrature points
        exp_c_theta : Array
            Exponential of the inner product between sufficient statistics and parameters
        dvol : Array
            Volume element at quadrature points

        Returns
        -------
        Array
            The D part of the Fisher metric, computed using quadrature weights
        """
        D = jnp.einsum('k,ki,kj,k', exp_c_theta.ravel(), c, c, dvol.ravel() * self._quadrature_weights)
        return D
