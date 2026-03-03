from abc import abstractmethod
from collections.abc import Callable
from functools import partial
from equinox import filter_jit
import jax
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module
from jax.lax import scan
from utils.vectorized import outer
import other_filter.resampling as resampling
from jax.lax import stop_gradient
from jaxtyping import Array, Float



class ExponentialFamily(Module):
    """
    A module representing an exponential family distribution with various statistical and geometric operations.

    This class provides functionality for working with exponential family distributions, including:
    - Computing natural and extended statistics
    - Calculating log partition functions and their derivatives
    - Computing Fisher information metrics
    - Computing Bergmann divergences
    - Performing numerical integration
    - Importance sampling
    - Computing cross entropy
    - Performing exponential map operations

    Attributes
    ----------
    _sample_space_dim : int
        Dimension of the sample space
    _bijection : Callable
        Function to transform coordinates
    _natural_statistics : Callable
        Function computing natural sufficient statistics
    _params_num : int
        Number of parameters
    _remaining_moments_num : int
        Number of remaining moments
    _remaining_moments : Callable
        Function computing remaining moments
    _extended_statistics : Callable
        Function computing extended statistics
    _bijection_params : tuple
        Parameters for bijection transformation
    _theta_indices_for_bijection_params : tuple
        Indices for mean and second moment parameters
    _indices_grid : tuple
        Grid of indices for computations
    _bijection_jac : Callable
        Jacobian of bijection transformation
    _stats_vect_sign : str
        Vectorization signature for statistics
    _par_vect_sign : str
        Vectorization signature for partition function
    log_dvolume : Callable
        Log volume element function
    log_partition : Callable
        Log partition function
    log_partition_jac : Callable
        Jacobian of log partition function
    log_partition_hess : Callable
        Hessian of log partition function
    log_partition_jac_hess : Callable
        Jacobian of Hessian of log partition function
    log_partition_extended : Callable
        Extended log partition function
    log_partition_extended_jac : Callable
        Jacobian of extended log partition function
    log_partition_extended_hess : Callable
        Hessian of extended log partition function
    """

    _sample_space_dim: int
    _bijection: Callable[[Array, tuple], Array]
    _natural_statistics: Callable[[Array], Array]
    _params_num: int
    _remaining_moments_num: int
    _remaining_moments: Callable[[Array, tuple], Array] = None
    _extended_statistics: Callable[[Array, tuple], Array]
    _bijection_params: tuple
    _theta_indices_for_bijection_params: tuple
    _indices_grid: tuple
    _bijection_jac: Callable[[Array, tuple], Array]
    _stats_vect_sign: str
    _par_vect_sign: str

    log_dvolume: Callable[[Array, tuple], Array]
    log_partition: Callable[[Array, tuple], Array]
    log_partition_jac: Callable[[Array, tuple], Array]
    log_partition_hess: Callable[[Array, tuple], Array]
    log_partition_jac_hess: Callable[[Array, tuple], Array]

    log_partition_extended: Callable[[Array, tuple], Array]
    log_partition_extended_jac: Callable[[Array, tuple], Array]
    log_partition_extended_hess: Callable[[Array, tuple], Array]

    epsilon: Float = jnp.finfo(jnp.float64).tiny

    def __init__(self,
                 sample_space_dimension: int,
                 bijection: Callable[[Array, tuple], Array],
                 statistics: Callable[[Array], Array],
                 remaining_statistics: [[Array], Array] = None,
                 bijection_parameters: tuple = None,
                 theta_indices_for_bijection_params: tuple = (jnp.array([0, ], dtype=jnp.int32),
                                                              jnp.array([1, ], dtype=jnp.int32)), ):
        """

        Parameters
        ----------
        sample_space_dimension
        bijection
        statistics
        remaining_statistics
        bijection_parameters
        theta_indices_for_bijection_params: tuple
            the first array contains the indices of mean,
            the second array contains the indices for the second moment
        """

        # d sample space dimension
        # m parameter space dimension
        bijection_vectorization_signature: str = '(d)->(d)'
        statistics_vectorization_signature: str = '(d)->(m)'
        partition_vectorization_signature: str = '(m)->()'

        self._theta_indices_for_bijection_params = theta_indices_for_bijection_params
        self._indices_grid = tuple(jnp.meshgrid(self._theta_indices_for_bijection_params[0],
                                                self._theta_indices_for_bijection_params[0], indexing='xy'))
        self._sample_space_dim = sample_space_dimension
        self._bijection_params = bijection_parameters
        self._bijection = jnp.vectorize(bijection, signature=bijection_vectorization_signature, excluded=(1,))

        # this returns the diagonal elements of Jacobian. There should be a better solution to this problem
        self._bijection_jac = filter_jit(jnp.vectorize(jax.jacobian(self._bijection), signature='(d)->(d,d)',
                                                excluded=(1,)))

        # dvolume is the absolute value of the jacobian determinant
        def _log_dvolume(_x, _params):
            jac = self._bijection_jac(_x, _params)
            _, log_det = jnp.linalg.slogdet(jac)
            return log_det

        self._stats_vect_sign = statistics_vectorization_signature
        self._par_vect_sign = partition_vectorization_signature
        self.log_dvolume = filter_jit(jnp.vectorize(_log_dvolume, signature=self._par_vect_sign, excluded=(1,)))
        self._natural_statistics = filter_jit(jnp.vectorize(statistics, signature=self._stats_vect_sign))

        temp = self._natural_statistics(jnp.zeros(self._sample_space_dim))

        self._params_num = temp.shape[0]
        self._remaining_moments_num = 0

        if remaining_statistics:
            self._remaining_moments = filter_jit(jnp.vectorize(remaining_statistics, signature=self._stats_vect_sign))

            def extended_statistics(x):
                return jnp.concatenate((self._natural_statistics(x), self._remaining_moments(x)))

            self._extended_statistics = filter_jit(jnp.vectorize(extended_statistics, signature=self._stats_vect_sign))

            if sample_space_dimension == 1:
                temp = self._remaining_moments(1.)
            else:
                temp = self._remaining_moments(jnp.ones(self._sample_space_dim))

            self._remaining_moments_num = temp.shape[0]
        else:
            self._extended_statistics = self._natural_statistics
            self._remaining_moments_num = 0

        def _log_partition(theta, bijection_params):
            # bijection params is considered as a constant, even if it might depend on theta
            bijection_params = stop_gradient(bijection_params)
            res, max_inner = self.integrate_partition(theta, bijection_params)
            return jnp.log(jnp.maximum(res,self.epsilon)) + max_inner

        def _log_partition_extended(theta_extended, bijection_params):
            """
            Formula 8.1
            """
            # bijection params is considered as a constant, even if it might depend on theta
            bijection_params = stop_gradient(bijection_params)
            res, max_inner = self.integrate_partition_extended(theta_extended, bijection_params)
            return jnp.log(jnp.maximum(res,self.epsilon)) + max_inner

        self.log_partition = filter_jit(jnp.vectorize(_log_partition, signature=self._par_vect_sign, excluded=(1,)))
        self.log_partition_jac = jax.jacobian(self.log_partition)
        self.log_partition_hess = jax.hessian(self.log_partition)
        self.log_partition_jac_hess = jax.jacobian(self.log_partition_hess)

        self.log_partition_extended = filter_jit(jnp.vectorize(_log_partition_extended,
                                                        signature=self._par_vect_sign, excluded=(1,)))
        self.log_partition_extended_jac = jax.jacobian(self.log_partition_extended)
        self.log_partition_extended_hess = jax.hessian(self.log_partition_extended)

    @property
    def sample_space_dim(self):
        return self._sample_space_dim

    @property
    def params_num(self):
        return self._params_num

    @property
    def remaining_moments_num(self):
        return self._remaining_moments_num

    @property
    def bijection_params(self):
        return self._bijection_params

    @bijection_params.setter
    def bijection_params(self, value):
        self._bijection_params = value

    @property
    def bijection(self):
        return self._bijection

    @property
    def natural_statistics(self):
        return self._natural_statistics

    @property
    def extended_statistics(self):
        return self._extended_statistics

    @property
    def higher_moments(self):
        return self._remaining_moments

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
    @filter_jit
    def natural_statistics_expectation(self, theta, bijection_params):
        return self.log_partition_jac(theta, bijection_params)

    @partial(jnp.vectorize, signature='(m)->(n)', excluded=[0, 2])
    @filter_jit
    def extended_statistics_expectation(self, theta, bijection_params):
        return self.log_partition_extended_jac(jnp.pad(theta, (0, self._remaining_moments_num)), bijection_params)

    @partial(jnp.vectorize, signature='(m)->(m,m)', excluded=[0, 2])
    @filter_jit
    def fisher_metric(self, theta, bijection_params):
        return self.log_partition_hess(theta, bijection_params)

    @partial(jnp.vectorize, signature='(m)->(m,m,m)', excluded=[0, 2])
    @filter_jit
    def double_inf_connection(self, theta, bijection_params):
        """
        Compute twice the information connection based on Amari's 1982 definition.

        The information connection (also called Amari-Chentsov tensor) is a tensor field that captures the geometric structure of a
        statistical manifold beyond the Fisher information metric. It measures how the Fisher information metric changes as we move
        in parameter space.

        For exponential families, it equals the third derivatives of the log partition function.

        Parameters
        ----------
        theta : Array
            Natural parameters of the exponential family distribution
        bijection_params : tuple
            Parameters defining the bijective transformation

        Returns
        -------
        Array
            A 3-dimensional array containing twice the information connection tensor, with shape (m,m,m) where m is the dimension of
            the parameter space
        """
        return self.log_partition_jac_hess(theta, bijection_params)

    @partial(jnp.vectorize, signature='(m),(m)->()', excluded=[0, 3])
    @filter_jit
    def bergmann_divergence(self, theta_1, theta_2, bijection_params):
        """
        Compute the Bergmann divergence between two parameters of an exponential family.

        The Bergmann divergence, also known as the Bregman divergence, is a measure of dissimilarity between two parameters
        of an exponential family distribution. It is defined as eq. (1.57) Amari 2016.:

        D(θ₁||θ₂) = ψ(θ₁) - ψ(θ₂) - ⟨η₂, θ₁ - θ₂⟩

        where:
        - ψ is the log partition function
        - η₂ is the expectation parameter corresponding to θ₂
        - ⟨·,·⟩ denotes the inner product

        The Bergmann divergence is always non-negative and equals zero if and only if θ₁ = θ₂.

        Parameters
        ----------
        theta_1 : Array
            First natural parameter vector
        theta_2 : Array
            Second natural parameter vector
        bijection_params : tuple
            Parameters for the bijective transformation

        Returns
        -------
        Array
            The Bergmann divergence D(θ₁||θ₂)
        """
        psi_1 = self.log_partition(theta_1, bijection_params)
        psi_2 = self.log_partition(theta_2, bijection_params)
        eta_2 = self.log_partition_jac(theta_2, bijection_params)

        return psi_1 - psi_2 - jnp.dot(eta_2, theta_1 - theta_2)

    @abstractmethod
    def get_density_values(self, grid_limits: Array, theta: Array, nb_of_points: Array,
                           bijection_params: tuple) -> \
            tuple[Array, Array]:
        """
        Get density values for the exponential family distribution.

        This function computes the probability density values on a specified grid for a given
        set of natural parameters theta.

        Parameters
        ----------
        grid_limits : Array
            Array with shape (N_d x 2) specifying the lower and upper limits of the grid for each dimension,
            where N_d is the number of dimensions
        theta : Array
            Natural parameters of the exponential family distribution
        nb_of_points : Array
            Array of integers specifying the number of grid points to use in each dimension
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        tuple[Array, Array]
            A tuple containing:
            - x_grid: The grid points where density values are computed
            - density: The probability density values at each grid point
        """
        raise NotImplementedError

    @abstractmethod
    def integrate_exponential_fun(self, log_fun: Callable[[Array, ], Array], bijection_params: tuple) \
            -> tuple[Array, Array]:
        """
        Integrate an exponential function using numerical integration techniques.

        This function performs numerical integration of an exponential function (e^{log_fun}) over the sample space of
        the exponential family distribution. It is used as a helper method for computing partition functions and
        related quantities.

        Parameters
        ----------
        log_fun : Callable[[Array, ], Array]
            A function that takes an array input and returns the logarithm of the function to be integrated
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        tuple[Array, Array]
            A tuple containing:
            - res: The value of the integral
            - max_inner: A scaling factor used to avoid numerical overflow during integration
        """
        raise NotImplementedError

    @abstractmethod
    def integrate(self, fun: Callable[[Array, ], Array], bijection_params: tuple) -> Array:
        """
        Integrate a function over the sample space of the exponential family.

        This function performs numerical integration of an arbitrary function over the sample space
        of the exponential family distribution.

        Parameters
        ----------
        fun : Callable[[Array, ], Array]
            A function to integrate that takes points from the sample space as input
            and returns scalar or vector values
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        Array
            The value of the integral ∫ fun(x)dx over the sample space
        """
        raise NotImplementedError

    def expected_value(self, statistic: Callable[[Array, ], Array], theta: Array,
                       bijection_params: tuple) -> Array:
        """
        Compute the expectation of a given statistic under an exponential family distribution.

        This method calculates E[statistic(X)] where X follows the exponential family distribution
        with natural parameter θ (theta). The expectation is computed numerically through
        integration.

        The probability density function used is:
            p_θ(x) = exp(⟨θ,c(x)⟩ - ψ(θ))
        where:
        - c(x) are the sufficient statistics
        - ψ(θ) is the log partition function
        - The integration is performed with respect to the reference measure

        Parameters
        ----------
        statistic : Callable[[Array, ], Array]
            The statistic function to compute expectation of, mapping from R^d to R^m
        theta : Array
            Natural parameters θ of the exponential family distribution
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        Array
            The expected value E[statistic(X)] where X follows the exponential family distribution
        """
        psi = self.log_partition(theta, bijection_params)

        def integrand(x: Array) -> Array:
            """
            Compute s(x)p_theta(x), s:R^d->R^m

            Parameters
            ----------
            x: ndarray
                vector from R^d

            Returns
            -------
            values
            """
            return statistic(x).T * jnp.exp(self.natural_statistics(x) @ theta - psi)

        return self.integrate(integrand, bijection_params)

    @abstractmethod
    def numerical_integration(self, numerical_values: Array, axis=None):
        raise NotImplementedError

    @abstractmethod
    def inner(self, theta: Array, bijection_params: tuple) -> Array:
        return NotImplementedError

    @abstractmethod
    def extended_inner(self, theta_extended: Array, bijection_params: tuple) -> Array:
        return NotImplementedError

    @filter_jit
    def integrate_partition(self, theta: Array, bijection_params: tuple) -> tuple[Array, Array]:
        """
        Integrate the exponential family partition function.

        This function computes the integral of exp(⟨θ,c(x)⟩) over the sample space, where:
        - θ is the natural parameter vector
        - c(x) are the sufficient statistics
        - The integration is performed with respect to the reference measure

        The partition function is a key quantity in exponential families as it normalizes
        the probability density and generates the moments of the distribution through its derivatives.
        This is where the children class need to implement their numerical integration to
        obtain the integration of log partition function at theta in R^d.
        Parameters
        ----------
        theta : Array
            Natural parameters of the exponential family distribution
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        tuple[Array, Array]
            A tuple containing:
            - res: The value of the partition function integral
            - max_inner: A scaling factor used to avoid numerical overflow during integration
        """
        inner_val = self.inner(theta, bijection_params)
        max_inner = jnp.max(inner_val)
        normalized_par_int = jnp.exp(inner_val - max_inner)

        res = self.numerical_integration(normalized_par_int)
        return res, max_inner

    @filter_jit
    def integrate_partition_extended(self, theta_extended: Array, bijection_params: tuple) -> \
            tuple[Array, Array]:
        """
        Integrate the extended partition function for the exponential family.
        This is where the children class need to implement their numerical integration to
        obtain the integration of log partition extended function at theta in R^d

        This function computes the integral of exp(⟨θ_ext,T_ext(x)⟩) over the sample space, where:
        - θ_ext is the extended natural parameter vector that includes both the natural parameters
          and parameters for the remaining statistics
        - T_ext(x) are the extended sufficient statistics that include both natural and remaining statistics
        - The integration is performed with respect to the reference measure

        The extended partition function is useful for computing expectations of the remaining statistics
        and for characterizing the full probabilistic structure of the exponential family beyond
        the minimal sufficient statistics.

        Parameters
        ----------
        theta_extended : Array
            Extended natural parameters, including both the natural parameters and parameters
            for the remaining statistics
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        tuple[Array, Array]
            A tuple containing:
            - res: The value of the extended partition function integral
            - max_inner: A scaling factor used to avoid numerical overflow during integration
        """
        inner_val = self.extended_inner(theta_extended, bijection_params)
        max_inner = jnp.max(inner_val)

        normalized_par_int = jnp.exp(inner_val - max_inner)
        res = self.numerical_integration(normalized_par_int)
        return res, max_inner

    @filter_jit
    def log_integrate(self,
                      log_fun: Callable[[Array, ], Array],
                      bijection_params: tuple) -> Array:
        """
        Compute the logarithm of an integral of an exponential function.

        This is a helper method that computes log(∫exp(log_fun(x))dx) while avoiding numerical overflow
        issues that can occur when computing exponential integrals directly. It uses the
        integrate_exponential_fun method which splits the computation into a normalized integral and
        a scaling factor.

        Parameters
        ----------
        log_fun : Callable[[Array, ], Array]
            Function that returns the log of the integrand at each point
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        Array
            The logarithm of the integral of exp(log_fun(x)) over the sample space
        """
        res, max_inner = self.integrate_exponential_fun(log_fun, bijection_params)
        return jnp.log(res) + max_inner

    @filter_jit
    def log_expected_value(self,
                           log_fun: Callable[[Array, ], Array],
                           theta: Array,
                           bijection_params: tuple) -> Array:
        """
        Compute the logarithm of the expected value of an exponential function under the exponential family distribution.

        This method calculates log(E[exp(log_fun(X))]) where X follows the exponential family distribution with
        natural parameter θ (theta). The expectation is computed through numerical integration while avoiding
        numerical overflow issues that can occur with exponential functions.

        The probability density function used is:
            p_θ(x) = exp(⟨θ,c(x)⟩ - ψ(θ))
        where:
        - c(x) are the sufficient statistics
        - ψ(θ) is the log partition function
        - The integration is performed with respect to the reference measure

        The computation is done in log space to maintain numerical stability when dealing with
        exponential functions that could otherwise cause overflow.

        Parameters
        ----------
        log_fun : Callable[[Array, ], Array]
            Function that returns the log of the statistic to compute expectation of
        theta : Array
            Natural parameters θ of the exponential family distribution
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space

        Returns
        -------
        Array
            The logarithm of E[exp(log_fun(X))] where X follows the exponential family distribution
        """
        psi = self.log_partition(theta, bijection_params)

        def integrand(x: Array) -> Array:
            """
            Compute s(x)p_theta(x), s:R^d->R^m

            Parameters
            ----------
            x: ndarray
                vector from R^d

            Returns
            -------
            values
            """
            return log_fun(x) + self.natural_statistics(x) @ theta - psi

        return self.log_integrate(integrand, bijection_params)

    @filter_jit
    def sample(self,
               shape: tuple,
               theta: Array,
               indices: Array,
               bijection_params: tuple,
               key: Array,
               ) -> Array:
        """
        Generate samples from an exponential family distribution using importance sampling.

        This method uses a multivariate Gaussian proposal distribution and systematic or stratified resampling
        to generate samples from the target exponential family distribution with given natural parameters θ.

        The proposal distribution is constructed using the mean vector (expected value of sufficient statistics)
        and covariance matrix (Fisher information metric) of the target distribution at θ.

        The importance weights are computed as the ratio of the target density to the proposal density (in log space),
        and resampling is performed using these weights to obtain the final samples.

        Parameters
        ----------
        shape : tuple
            Tuple specifying the desired shape of the output samples
        theta : Array
            Natural parameters θ of the target exponential family distribution
        indices : Array
            Indices for selecting components of the natural statistics corresponding to sample space dimensions
        bijection_params : tuple
            Parameters defining the bijective transformation applied to the sample space
        key : Array
            Random number generator key (JAX PRNGKey) for generating random samples

        Returns
        -------
        Array
            Array of samples from the target distribution with shape (1, count, d) where:
            - count is the number of samples specified in shape
            - d is the dimension of the sample space
        """

        # fist get a Gaussian samples from gaussian density with mean and variance according to
        # p_\theta mean and variance
        eta = self.natural_statistics_expectation(theta, bijection_params)
        g = self.fisher_metric(theta, bijection_params)
        indices_grid = tuple(jnp.meshgrid(indices,
                                          indices, indexing='xy'))

        mean = jnp.take(eta, indices)
        cov = g[indices_grid]

        key, a_key = jrandom.split(key)
        gaussian_samples = jrandom.multivariate_normal(key, mean, cov, shape)
        inner = self._natural_statistics(gaussian_samples) @ theta
        max_inner = jnp.max(inner)
        log_exponential_density_un_normalized = inner - max_inner
        log_gaussian = resampling.log_gaussian_density(gaussian_samples - mean, cov)
        log_weight = resampling.normalize_log_weights(log_exponential_density_un_normalized - log_gaussian)

        uni = jrandom.uniform(key)

        samples = resampling.systematic_or_stratified(gaussian_samples, log_weight, uni)

        return samples

    @filter_jit
    def cross_entropy(self,
                      samples: Array,
                      theta: Array,
                      bijection_params: tuple) -> Float:
        """
        Calculate the cross entropy between an empirical distribution and an exponential family distribution.

        This method computes H(q||p_θ) where:
        - q is the empirical distribution represented by the given samples
        - p_θ is the exponential family distribution with natural parameter θ
        - H(q||p_θ) = -E_q[log p_θ]

        The cross entropy measures how different q is from p_θ. Lower values indicate better fit between
        the empirical and exponential family distributions.

        For exponential families, the cross entropy has the form:
            H(q||p_θ) = -⟨θ,E_q[c(X)]⟩ + ψ(θ)
        where:
        - c(X) are the sufficient statistics
        - ψ(θ) is the log partition function
        - E_q denotes expectation under empirical distribution q

        Parameters
        ----------
        samples : Array
            Array of samples drawn from empirical distribution q
        theta : Array
            Natural parameters θ of exponential family distribution p_θ
        bijection_params : tuple
            Parameters defining bijective transformation of sample space

        Returns
        -------
        Float
            The cross entropy H(q||p_θ) between empirical distribution q and
            exponential family distribution p_θ
        """
        psi = self.log_partition(theta, bijection_params)
        # N = samples.shape[0]  # sample size
        result = - jnp.mean((self._natural_statistics(samples) @ theta - psi))

        return result

    def exponential_map(self, theta: Array, d_theta: Array,
                        bijection_params: tuple,
                        steps: int) -> Array:
        """
        Compute the Riemannian exponential map for an exponential family distribution.

        The exponential map maps a tangent vector at a point in the statistical manifold to another point
        on the manifold by following the geodesic in the direction of that vector.

        For exponential families, the geodesic equations can be solved numerically using a discretized ODE solver.
        This function implements a discrete version of the exponential map by solving the geodesic ODEs
        using numerical integration over a specified number of steps.

        The geodesic equations for exponential families are derived from the Fisher-Rao metric tensor,
        which provides a natural Riemannian structure to the statistical manifold. The solution gives
        the path of steepest descent/ascent in the parameter space.

        This can be used for optimization on the statistical manifold or for interpolating between
        probability distributions in a geometrically natural way.

        Parameters
        ----------
        theta : Array
            Starting point (natural parameters) on the statistical manifold
        d_theta : Array
            Tangent vector at theta specifying the direction and speed of the geodesic
        bijection_params : tuple
            Parameters for the bijective transformation of the sample space
        steps : int
            Number of integration steps for solving the geodesic equations

        Returns
        -------
        Array
            The endpoint of the geodesic after following it for time t_max, where t_max is
            determined by the Riemannian norm of d_theta
        """
        t_max = self.riemannian_norm(theta, d_theta)
        t = jnp.linspace(0, t_max, steps, endpoint=True)
        dt = t_max / steps

        @filter_jit
        def integrator_loop(carry_, inputs_):
            # TODO: implement Geodesic update here via RK4
            # the ode is d^2theta
            theta_, d_theta_, params_, eta_tilde_, fisher_ = carry_

            return carry_, carry_

        eta_tilde = self.extended_statistics_expectation(theta, bijection_params)
        fisher = self.fisher_metric(theta, bijection_params)
        theta_end, _ = scan(integrator_loop, (theta, d_theta, bijection_params, eta_tilde, fisher)
                            , (t,))

        return theta_end

    def riemannian_inner_product(self, theta, v, w):
        """
        Calculate the Riemannian inner product between two tangent vectors at a point on the statistical manifold.

        The Riemannian inner product defines a smoothly varying inner product on the tangent spaces of the statistical
        manifold. For exponential families, it is given by:

        ⟨v,w⟩_θ = 0.25 * v^T I(θ) w

        where:
        - I(θ) is the Fisher information metric at θ
        - v,w are tangent vectors at θ
        - The factor 0.25 comes from working in the √p parameterization

        This inner product is fundamental for defining geometric quantities like angles and lengths on the statistical
        manifold. It induces a Riemannian metric structure that respects the statistical properties of the manifold.

        Parameters
        ----------
        theta : Array
            Point on the statistical manifold (natural parameters)
        v : Array
            First tangent vector at theta
        w : Array
            Second tangent vector at theta

        Returns
        -------
        Float
            The Riemannian inner product ⟨v,w⟩_θ between the tangent vectors
        """
        return 0.25 * v @ (self.fisher_metric(theta) @ w)

    def riemannian_norm(self, theta, v):
        """
        Calculate the Riemannian norm of a tangent vector at a point on the statistical manifold.

        The Riemannian norm of a tangent vector v at a point θ is defined as:

        ||v||_θ = √⟨v,v⟩_θ

        where ⟨·,·⟩_θ is the Riemannian inner product at θ. For exponential families, this equals:

        ||v||_θ = √(0.25 * v^T I(θ) v)

        where I(θ) is the Fisher information metric at θ.

        The Riemannian norm measures the length of a tangent vector in the curved geometry of the
        statistical manifold. It is used to measure distances and compute geometric quantities on
        the manifold.

        Parameters
        ----------
        theta : Array
            Point on the statistical manifold (natural parameters)
        v : Array
            Tangent vector at theta whose norm is to be computed

        Returns
        -------
        Float
            The Riemannian norm ||v||_θ of the tangent vector v at point θ
        """

        return jnp.sqrt(self.riemannian_inner_product(theta, v, v))

    @filter_jit
    def update_bijection_params(self, eta: Array, fisher: Array, old_bijection_params: tuple):
        """
        Update the bijection parameters of an exponential family distribution using expected values and Fisher information.

        This method updates the parameters that define the bijective transformation of the sample space
        based on the current expected values (eta) and Fisher information matrix (fisher) of the distribution.
        It extracts the mean vector and covariance matrix from eta and fisher using pre-specified indices,
        while preserving the scale factor from the old parameters.

        Parameters
        ----------
        eta : Array
            Vector of expected values (mean parameters) of the sufficient statistics
        fisher : Array
            Fisher information matrix at the current parameter values
        old_bijection_params : tuple
            Current bijection parameters containing (mean, covariance, scale_factor)

        Returns
        -------
        tuple
            Updated bijection parameters (new_mean, new_covariance, old_scale_factor)
        """
        _mu = jnp.take(eta, self._theta_indices_for_bijection_params[0])
        _Sigma = fisher[self._indices_grid]

        # do not update scale factor
        _, _, scale_factor = old_bijection_params

        new_bijection_params = _mu, _Sigma, scale_factor
        return new_bijection_params

    @filter_jit
    def update_chol_bijection_params(self, eta: Array, fisher: Array, old_bijection_params: tuple):
        """
        Update the bijection parameters of an exponential family distribution using expected values and Fisher information.

        This method updates the parameters that define the bijective transformation of the sample space
        based on the current expected values (eta) and Fisher information matrix (fisher) of the distribution.
        It extracts the mean vector and covariance matrix from eta and fisher using pre-specified indices,
        while preserving the scale factor from the old parameters.

        Parameters
        ----------
        eta : Array
            Vector of expected values (mean parameters) of the sufficient statistics
        fisher : Array
            Fisher information matrix at the current parameter values
        old_bijection_params : tuple
            Current bijection parameters containing (mean, covariance, scale_factor)

        Returns
        -------
        tuple
            Updated bijection parameters (new_mean, new_covariance, old_scale_factor)
        """
        _mu = jnp.take(eta, self._theta_indices_for_bijection_params[0])
        _Sigma = eta[self._theta_indices_for_bijection_params[1]] - jnp.outer(_mu, _mu)

        #conditioning
        min_ev_Sigma = jnp.min(jnp.linalg.eigvalsh(_Sigma))
        epsilon = jnp.where(min_ev_Sigma>0, 0, jnp.maximum(1e-10, jnp.abs(min_ev_Sigma)))
        _Sigma = _Sigma + 2 * epsilon * jnp.eye(_Sigma.shape[0])  # conditioning

        # do not update scale factor
        _, _, scale_factor = old_bijection_params

        new_bijection_params = _mu, jnp.linalg.cholesky(_Sigma), scale_factor
        return new_bijection_params

    @filter_jit
    def update_bijection_params_vanilla(self, eta: Array, fisher: Array, old_bijection_params: tuple):
        """
        Update the bijection parameters of an exponential family distribution using a simpler approach.

        This method modifies the parameters that define the bijective transformation of the sample space
        using the current expected values (eta), but unlike the standard update method, it calculates
        the covariance directly from the first and second moments rather than using the Fisher information.

        The method:
        1. Extracts the mean vector from eta using pre-specified indices
        2. Computes covariance as second moment minus outer product of means
        3. Preserves the scale factor from the old parameters

        Parameters
        ----------
        eta : Array
            Vector of expected values (mean parameters) of the sufficient statistics
        fisher : Array
            Fisher information matrix (unused in this vanilla version)
        old_bijection_params : tuple
            Current bijection parameters containing (mean, covariance, scale_factor)

        Returns
        -------
        tuple
            Updated bijection parameters (new_mean, new_covariance, old_scale_factor)
        """
        _mu = jnp.take(eta, self._theta_indices_for_bijection_params[0])
        # _Sigma = fisher[self._indices_grid]
        _Sigma = eta[self._theta_indices_for_bijection_params[1]] - jnp.outer(_mu, _mu)

        # do not update scale factor
        _, _, scale_factor = old_bijection_params
        new_bijection_params = _mu, _Sigma, scale_factor
        return new_bijection_params

    @filter_jit
    def fisher_metric_from_samples(self, samples: Array)-> Array:
        # samples of the shape (N_devices, N_particle_per_device, N_dim)

        sampled_nat_stats = self.natural_statistics(samples)
        mean_nat_stats = jnp.mean(sampled_nat_stats,axis=[0,1])
        eta_tilde_sampled = (sampled_nat_stats - mean_nat_stats)
        return jnp.mean(outer(eta_tilde_sampled,eta_tilde_sampled),axis=[0,1])

    @filter_jit
    def expected_values_from_samples(self, statistic: Callable[[Array, ], Array],
                                     samples: Array)-> Array:
        # samples of the shape (N_devices, N_particle_per_device, N_dim)
        return jnp.mean(statistic(samples),axis=[0,1])
