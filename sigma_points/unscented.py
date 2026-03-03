from sigma_points.sigma_points import SigmaPoints
import jax.numpy as jnp
import numpy as onp
from equinox import filter_jit
from jaxtyping import Array


class UnscentedSigmaPoints(SigmaPoints):
    """Generate weights and Sigma points according to the scaled unscented transformation.

    The UnscentedSigmaPoints class generates Sigma points (sampling points) and weights according
    to the scaled unscented transformation. This is done by applying a deterministic sampling
    scheme that captures the mean and covariance of a probability distribution.

    Parameters
    ----------
    n_state : int
        Dimension of the state vector
    kappa : float
        Secondary scaling parameter, usually 0
    alpha : float
        Primary scaling parameter determining how far Sigma points are spread (usually 1e-3)
    beta : float
        Parameter for incorporating prior knowledge of state distribution (2 is optimal for Gaussians)

    Attributes
    ----------
    lamb : float
        Scaling parameter lambda calculated as (n + kappa) * alpha^2 - n
    weights_mean : Array
        Weights used for mean calculation
    weights_cov : Array
        Weights used for covariance calculation
    native_points : Array
        Base sigma points before scaling

    Methods
    -------
    get_sigma_points(mean, cov)
        Generates the Sigma points given a mean and covariance
    """
    kappa: float
    alpha: float
    beta: float
    lamb: float

    def __init__(self, n_state: int, kappa: float, alpha: float, beta: float):
        self.n_state = n_state
        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta
        self.lamb = (self.n_state + self.kappa) * (self.alpha ** 2) - self.n_state

        # populate weights
        # see https://users.aalto.fi/~ssarkka/course_k2016/handout5.pdf
        # wm is W_i^(m), wp is W_i^(c)
        wm = onp.full(2 * self.n_state + 1, 1 / (2 * (self.n_state + self.lamb)))
        wp = onp.copy(wm)

        wm[0] = self.lamb / (self.n_state + self.lamb)
        wp[0] = self.lamb / (self.n_state + self.lamb) + (1 - self.alpha ** 2 + self.beta)

        self.weights_mean = jnp.asarray(wm)
        self.weights_cov = jnp.asarray(wp)

        # calculate xi
        self.native_points = jnp.vstack((jnp.zeros((1, self.n_state)), jnp.eye(self.n_state), -jnp.eye(self.n_state)))

    @filter_jit
    def get_sigma_points(self, mean: Array, cov: Array):
        scaled_sqrt_cov = jnp.sqrt(self.n_state + self.lamb) * jnp.linalg.cholesky(cov)
        return mean + self.native_points @ scaled_sqrt_cov.T
