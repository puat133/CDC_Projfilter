import abc
from jaxtyping import Array
from equinox import Module


class SigmaPoints(Module):
    """
    Base class for sigma point calculation methods.

    A sigma point is a deterministic sampling of a probability distribution,
    used in Unscented Kalman Filters and similar algorithms. This class provides
    the interface for different sigma point calculation schemes.

    Parameters
    ----------
    n_state : int
        Dimension of the state space.
    weights_mean : Array
        Array of mean weights for computing state mean.
    weights_cov : Array
        Array of covariance weights for computing state covariance.
    native_points : Array
        Array of base sigma points in canonical coordinates.

    Notes
    -----
    This is an abstract base class that should be subclassed to implement
    specific sigma point calculation methods.
    """
    n_state: int
    weights_mean: Array
    weights_cov: Array
    native_points: Array

    @abc.abstractmethod
    def get_sigma_points(self, mean: Array, cov: Array):
        raise NotImplementedError
