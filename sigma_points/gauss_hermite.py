import math

import jax.numpy as jnp
from equinox import filter_jit
from scipy.special import roots_hermitenorm, eval_hermitenorm

from sigma_points.sigma_points import SigmaPoints


class GaussHermiteSigmaPoints(SigmaPoints):
    """Gauss-Hermite Sigma Points

    A class to generate sigma points and weights based on the Gauss-Hermite
    quadrature rule.

    Parameters
    ----------
    n_state : int
        Number of state dimensions
    order : int, optional
        Order of Gauss-Hermite quadrature (default is 3)

    Attributes
    ----------
    weights_mean : jnp.ndarray
        Weights for computing mean
    weights_cov : jnp.ndarray
        Weights for computing covariance
    native_points : jnp.ndarray
        Unweighted sigma points

    Notes
    -----
    The Gauss-Hermite sigma point method uses Gauss-Hermite quadrature to approximate
    the nonlinear transformation of a Gaussian random variable.
    """
    order: int

    def __init__(self, n_state: int, order: int = 3):
        self.n_state = n_state
        self.order = order

        roots, _ = roots_hermitenorm(self.order)
        eval_hermite_order_min_1_at_roots = eval_hermitenorm(order - 1, roots)
        w_1d = math.factorial(self.order) / (eval_hermite_order_min_1_at_roots ** 2 * self.order ** 2)
        x_ = []

        for i in range(self.n_state):
            x_.append(roots)

        grids = jnp.meshgrid(*x_, indexing='xy')
        grids = jnp.stack(grids, axis=-1)
        xi = grids.reshape((roots.shape[0] ** self.n_state, self.n_state))

        w = w_1d
        for i in range(self.n_state - 1):
            w = jnp.kron(w, w_1d)
        w = w/jnp.sum(w)

        self.weights_mean = jnp.asarray(w)
        self.weights_cov = jnp.asarray(w)

        self.native_points = xi

    @filter_jit
    def get_sigma_points(self, mean: jnp.ndarray, cov: jnp.ndarray):
        sqrt_cov = jnp.linalg.cholesky(cov)
        return mean + self.native_points @ sqrt_cov.T


def get_hermite_coeff(order: int) -> list:
    """
    Generate coefficients for probabilist Hermite polynomials from order 0 to p.

    The probabilist Hermite polynomials differ from the physicist Hermite polynomials in
    their normalization. This function computes the coefficients for each order,
    with coefficients ordered from highest to lowest degree.

    Parameters
    ----------
    order : int
        Maximum order of Hermite polynomials to generate (p)

    Returns
    -------
    list
        List containing arrays of coefficients for orders 0 to p.
        Each array contains coefficients ordered from highest to lowest degree.

    Notes
    -----
    The probabilist Hermite polynomials are defined by the recurrence relation:
    H_{n+1}(x) = x H_n(x) - n H_{n-1}(x)
    with H_0(x) = 1 and H_1(x) = x
    """
    H0 = jnp.array([1])
    H1 = jnp.array([1, 0])

    H = [H0, H1]

    for i in range(2, order + 1):
        H.append(jnp.append(H[i - 1], 0) -
                 (i - 1) * jnp.pad(H[i - 2], (2, 0), 'constant', constant_values=0))

    return H
