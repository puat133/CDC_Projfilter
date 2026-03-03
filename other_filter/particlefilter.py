from collections.abc import Callable
from functools import partial
from equinox import filter_jit
import jax.numpy as jnp
import jax.random as jrandom
from equinox import Module
from jax import lax, pmap
from jaxtyping import Array
from other_filter.resampling import log_mean_exp, systematic_or_stratified


# The jitted function ___ includes a pmap. Using jit-of-pmap can lead to inefficient data
# movement, as the outer jit does not preserve sharded data representations and instead collects input and output
# arrays onto a single device. Consider removing the outer jit unless you know what you're doing

class ParticleFilter(Module):
    """A particle filter implementation for state estimation.

    This particle filter supports parallel processing across multiple devices using JAX's pmap.

    Parameters
    ----------
    n_devices : int
        Number of parallel devices to use.
    n_particle_per_device : int
        Number of particles per device.
    initial_samples : Array
        Initial state samples.
    measurement_history : Array
        History of measurements.
    transition_fun : callable
        State transition function.
    neg_log_likelihood : callable
        Negative log likelihood function.
    prng_key : Array
        Random number generator key.

    Attributes
    ----------
    n_devices : int
        Number of parallel devices.
    n_state : int
        Dimension of state vector.
    n_particle_per_device : int
        Number of particles per device.
    initial_samples : Array
        Initial state samples.
    mean_init : Array
        Mean of initial samples.
    measurement_history : Array
        History of measurements.
    transition_fun : callable
        State transition function.
    neg_log_likelihood : callable
        Negative log likelihood function.
    prng_key : Array
        Random number generator key.
    uniforms : Array
        Uniform random numbers.
    particle_filter_body : callable
        Main body of particle filter algorithm.
    """
    n_devices: int
    n_devices: int
    n_state: int
    n_particle_per_device: int
    initial_samples: Array
    mean_init: Array
    measurement_history: Array
    transition_fun: Callable[[Array], Array]
    neg_log_likelihood: Callable[[Array, Array], Array]
    measurement_history: Array
    prng_key: Array
    uniforms: Array
    particle_filter_body: Callable[[tuple, tuple], [tuple, tuple]]

    def __init__(self,
                 n_devices: int,
                 n_particle_per_device: int,
                 initial_samples: Array,
                 measurement_history: Array,
                 transition_fun: Callable[[Array], Array],
                 neg_log_likelihood: Callable[[Array, Array], Array],
                 prng_key: Array):
        self.transition_fun = transition_fun
        self.neg_log_likelihood = neg_log_likelihood
        self.measurement_history = measurement_history
        self.initial_samples = initial_samples
        self.mean_init = jnp.mean(self.initial_samples, axis=(0, 1))
        self.prng_key = prng_key
        self.n_devices = n_devices
        self.n_particle_per_device = n_particle_per_device
        self.n_state = initial_samples.shape[-1]
        self.prng_key, subkey = jrandom.split(self.prng_key)
        self.uniforms = jrandom.uniform(subkey, (self.measurement_history.shape[0],))

        @filter_jit
        @partial(jnp.vectorize, signature='(n),(n)->(n),()', excluded=[2])
        def _parallelized_routine(x_particle_, q_particle_, meas):
            x_particle_ = self._transition_fun(x_particle_)
            x_particle_ += q_particle_
            log_weights_ = - neg_log_likelihood(x_particle_, meas)
            return x_particle_, log_weights_

        # @filter_jit
        def _particle_filter_body(carry_: tuple, inputs_: tuple):
            x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            y_, uni_ = inputs_
            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            q_particle_ = self._sqrt_diag_process_cov * jrandom.normal(subkey_, (
                self._n_devices,
                self._n_particle_per_device,
                self._mean_init.shape[-1]))
            x_particle_, log_weights_ = pmap(_parallelized_routine,
                                             axis_name='device_axis')(x_particle_resampled_,
                                                                      q_particle_,
                                                                      jnp.tile(y_, [self._n_devices, 1]))

            x_particle_resampled_ = systematic_or_stratified(x_particle_, log_weights_, uni_)
            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.

            neg_likelihood_ -= log_mean_exp(log_weights_.ravel())
            return (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_)

        self.particle_filter_body = _particle_filter_body

    def run(self) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Run the particle filter.

        Returns
        -------
        neg_log_likelihood_end: Array
            Final negative log likelihood value.
        state_end: Array
            Final estimated state.
        x_particle_history: Array
            History of particle states over time.
        neg_likelihood_history: Array
            History of negative log likelihood values.
        log_weights_history: Array
            History of particle log weights.
        estimated_state_history: Array
            History of estimated states over time.
        """
        # no need to be normalized
        log_weigths_init = jnp.zeros((self.n_devices, self.n_particle_per_device))
        neg_likelihood_init = 0
        carry = (self.initial_samples, self.mean_init,
                 log_weigths_init, neg_likelihood_init,
                 self.prng_key)
        inputs = (self.measurement_history, self.uniforms)

        (x_particle_end, state_end,
         log_weights_end, neg_log_likelihood_end,
         a_prng_key), res = lax.scan(self.particle_filter_body,
                                         carry,
                                         inputs)

        x_particle_history, log_weights_history, estimated_state_history, neg_likelihood_history = res

        return neg_log_likelihood_end, state_end, x_particle_history, neg_likelihood_history, \
            log_weights_history, estimated_state_history
