from collections.abc import Callable

import jax.numpy as jnp
from equinox import filter_jit
from sparse_quadrature.curtis_clenshaw import sparse_clenshaw_curtis_quadrature
from sparse_quadrature.kronrod import sparse_kronrod_quadrature
from sparse_quadrature.patterson import sparse_patterson_quadrature
from sparse_quadrature.gauss_legendre import sparse_gauss_legendre_quadrature
from exponential_family.n_d_ef_sparse import NDExponentialFamilySparse
from jaxtyping import Array, Float

def plain_gauss_bijection_original(xtilde: Array, bijection_params: tuple):
    """
    Apply Gaussian bijection to quadrature points.

    Parameters
    ----------
    xtilde : Array
        Array of transformed quadrature points from hypercube [-1,1]^d where each point is erfinv transformed
    bijection_params : tuple
        Tuple containing (mu, Sigma, scale_factor) where:
            mu : Array - Mean vector
            Sigma : Array - Covariance matrix
            scale_factor : Float - Scaling factor for the transformation

    Returns
    -------
    Array
        Transformed points after applying Gaussian bijection
    """

    # here we assume that xtilde is erfinv(x), where x is a quadrature point from hypercube [-1,1]^d
    # we have adjusted the quadrature node weights accordingly in n_d_ef_sparse.



    _mu, _Sigma, scale_factor = bijection_params

    # the T in eq (20) of Gaussian-Based Parametric Bijections For Automatic Projection Filters is infact inverse
    # of the eigenvectors. So we do not invert it here.

    eig_vals, eig_vects = jnp.linalg.eigh(_Sigma)
    return _mu + scale_factor * jnp.sqrt(2) * (eig_vects @ (jnp.sqrt(eig_vals)*xtilde))

def plain_gauss_bijection(xtilde: Array, bijection_params: tuple):
    """
    Apply Gaussian bijection to quadrature points.

    Parameters
    ----------
    xtilde : Array
        Array of transformed quadrature points from hypercube [-1,1]^d where each point is erfinv transformed
    bijection_params : tuple
        Tuple containing (mu, Sigma, scale_factor) where:
            mu : Array - Mean vector
            Sigma : Array - Covariance matrix
            scale_factor : Float - Scaling factor for the transformation

    Returns
    -------
    Array
        Transformed points after applying Gaussian bijection
    """

    # here we assume that xtilde is erfinv(x), where x is a quadrature point from hypercube [-1,1]^d
    # we have adjusted the quadrature node weights accordingly in n_d_ef_sparse.

    _mu, _Sigma, scale_factor = bijection_params
    return _mu + scale_factor * jnp.sqrt(2) * jnp.linalg.cholesky(_Sigma) @ xtilde


def plain_gauss_bijection_chol(xtilde: Array, bijection_params: tuple):
    """
    Apply Gaussian bijection to quadrature points using pre-computed Cholesky decomposition.

    Parameters
    ----------
    xtilde : Array
        Array of transformed quadrature points from hypercube [-1,1]^d where each point is erfinv transformed
    bijection_params : tuple
        Tuple containing (mu, Sigma_chol, scale_factor) where:
            mu : Array - Mean vector
            Sigma_chol : Array - Pre-computed Cholesky factor of covariance matrix
            scale_factor : Float - Scaling factor for the transformation

    Returns
    -------
    Array
        Transformed points after applying Gaussian bijection using Cholesky decomposition
    """
    # here we assume that xtilde is erfinv(x), where x is a quadrature point from hypercube [-1,1]^d
    # we have adjusted the quadrature node weights accordingly in n_d_ef_sparse.

    _mu, _Sigma_chol, scale_factor = bijection_params
    return _mu + scale_factor * jnp.sqrt(2) * _Sigma_chol @ xtilde


gauss_bijection = filter_jit(jnp.vectorize(plain_gauss_bijection, signature='(n)->(n)', excluded=(1,)))
gauss_bijection_chol = filter_jit(jnp.vectorize(plain_gauss_bijection_chol, signature='(n)->(n)', excluded=(1,)))
gauss_bijection_original = filter_jit(jnp.vectorize(plain_gauss_bijection_original, signature='(n)->(n)', excluded=(1,)))


class NDExponentialFamilySPG(NDExponentialFamilySparse):
    """
    N-dimensional exponential family with sparse grids.

    This class implements an N-dimensional exponential family distribution using sparse grid quadrature for
    numerical integration. The log partition function and related computations are performed using sparse
    grid quadrature points, allowing for efficient high-dimensional integration.

    The sparse grid points are generated using one of four available quadrature rules:
    - Clenshaw-Curtis (default)
    - Gauss-Kronrod
    - Gauss-Patterson
    - Gauss-Legendre

    The bijection parameter allows transformation of the integration domain to match the support of the
    target distribution. The statistics function defines the sufficient statistics of the exponential family.

    Parameters
    ----------
    sample_space_dimension : int
        Dimension of the sample space d ≥ 1
    sparse_grid_level : int
        Level of sparse grid approximation
    bijection : callable
        Function to transform sparse grid points to target domain
    statistics : callable
        Function computing sufficient statistics
    remaining_statistics : callable, optional
        Additional statistics not used in exponential family
    epsilon : Float, optional
        Small constant for numerical stability, default 1e-7
    s_rule : str, optional
        Sparse grid rule: "clenshaw-curtis", "gauss-kronrod", "gauss-patterson", or "gauss-legendre"
    bijection_parameters : tuple, optional
        Parameters for bijection transformation
    theta_indices_for_bijection_params : tuple, optional
        Indices of parameters used in bijection
    direct_calculation : bool, optional
        Whether to use direct calculation method, default False
    Exponential family for sample space with dimension `d` >=1, and parameter space with dimension `m` where
    of log partition function is solved via `n` Quasi Monte Carlo points (Halton low discrepancy points).
    """

    def __init__(self,
                 sample_space_dimension: int,
                 sparse_grid_level: int,
                 bijection: Callable[[Array, tuple], Array],
                 statistics: Callable[[Array], Array],
                 remaining_statistics: [[Array], Array] = None,
                 epsilon: Float = 1e-7,
                 s_rule: str = "gauss-patterson",
                 bijection_parameters: tuple = None,
                 theta_indices_for_bijection_params: tuple = (jnp.array([0, ], dtype=jnp.int32),
                                                              jnp.array([1, ], dtype=jnp.int32)),
                 direct_calculation: bool = False
                 ):


        if s_rule.lower() in ["clenshaw-curtis", "gauss-kronrod", "gauss-legendre", "gauss-patterson"]:
            if s_rule == "gauss-patterson" and sparse_grid_level > 8:
                sparse_grid_level = 8
            s_rule = s_rule
        else:
            s_rule = "clenshaw-curtis"

        super().__init__(sample_space_dimension=sample_space_dimension,
                         sparse_grid_level=sparse_grid_level,
                         bijection=bijection,
                         statistics=statistics,
                         remaining_statistics=remaining_statistics,
                         bijection_parameters=bijection_parameters,
                         epsilon=epsilon,
                         s_rule=s_rule,
                         theta_indices_for_bijection_params=theta_indices_for_bijection_params,
                         direct_calculation=direct_calculation)

    def initialize_sparse_grid(self):
        # check if sparse grid level is within valid range for each quadrature rule
        # Gauss-Patterson quadrature is only defined for levels 0 through 8
        if self._s_rule.lower() == "gauss-patterson" and self._spg_level > 8:
            self._spg_level = 8

        if self._s_rule.lower() == "gauss-kronrod":
            _, points, weights = sparse_kronrod_quadrature(self._sample_space_dim,
                                                           self._spg_level)
        elif self._s_rule.lower() == "gauss-patterson":
            _, points, weights = sparse_patterson_quadrature(self._sample_space_dim,
                                                             self._spg_level)
        elif self._s_rule.lower() == "gauss-legendre":
            _, points, weights = sparse_gauss_legendre_quadrature(self._sample_space_dim,
                                                                  self._spg_level)
        else:
            _, points, weights = sparse_clenshaw_curtis_quadrature(self._sample_space_dim,
                                                                   self._spg_level)
        return points, weights
