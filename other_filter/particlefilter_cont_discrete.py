from collections.abc import Callable
from typing import Tuple, Any
from jax import jacobian
import diffrax as dfx
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
from equinox import filter_jit
from jax import lax
from datetime import datetime, timedelta
from other_filter.particlefilter import ParticleFilter
from other_filter.resampling import log_mean_exp, systematic_or_stratified, multinomial
from other_filter.resampling import normalize_log_weights
from utils.diffrax import VectorizedControlTerm
from jaxtyping import Array, PyTree

class ContinuousDiscreteParticleFilter(ParticleFilter):
    """A class for continuous-discrete particle filtering.

    This implements a particle filter for systems with continuous-time dynamics
    and discrete-time measurements. The state evolution is modeled by a
    stochastic differential equation (SDE) and measurements arrive at discrete
    time intervals.

    Parameters
    ----------
    n_devices : int
        Number of parallel devices/cores to use
    n_particle_per_device : int
        Number of particles per device
    initial_samples : ndarray
        Initial particle state samples
    measurement_history : ndarray
        Array of measurement data
    process_drift : callable
        Drift function f(t,x) for the SDE dx = f(t,x)dt + g(t,x)dW
    process_diffusion : callable
        Diffusion function g(t,x) for the SDE
    process_brownian_dim : int
        Dimension of the Brownian motion
    dt : float
        Time step for SDE solver
    dt_meas : float
        Time interval between measurements
    resampling : str, optional
        Resampling scheme to use ('systematic', 'stratified', 'multinomial')
    sde_solver : AbstractSolver, optional
        SDE solver to use (default: EulerHeun)

    Methods
    -------
    run()
        Run the particle filter and return results
    _sde_propagation_and_log_weights(x_particle_initial, t_s, meas, brownian_motion)
        Propagate particles using SDE and compute log weights
    """
    process_drift: Callable[[float, Array, PyTree], Array]
    process_diffusion: Callable[[float, Array, PyTree], Array]
    process_brownian_dim: int
    dt: float
    dt_meas: float
    resampling: str
    sde_solver: dfx.AbstractSolver

    def __init__(self,
                 n_devices: int,
                 n_particle_per_device: int,
                 initial_samples: Array,
                 measurement_history: Array,
                 process_drift: Callable[[float, Array, PyTree], Array],
                 process_diffusion: Callable[[float, Array, PyTree], Array],
                 negative_likelihood: Callable[[Array, Array], Array],
                 process_brownian_dim: int,
                 dt: float,
                 dt_meas: float,
                 prng_key: Array,
                 resampling='systematic',
                 sde_solver: dfx.AbstractSolver = dfx.ShARK(), #only valid for Additive Noise
                 use_stratonovich: bool = True,):

        process_diffusion_jac = jacobian(process_diffusion, argnums=1)
        # convert the sde to stratonovich form according to
        # eq. 5.3 Theory and Numerics of Differential Equations: Durham
        # https://books.google.nl/books?id=_WfwCAAAQBAJ&pg=PA137&lpg=PA137#v=onepage&q&f=false

        @filter_jit
        @partial(jnp.vectorize, signature='(n)->(n)', excluded=(0, 2))
        def process_drift_stratonovich(_t:float, _x:Array, args:PyTree):
            f_eval = process_drift(_t, _x, args)
            g_eval = process_diffusion(_t, _x, args)
            dg_dx_eval = process_diffusion_jac(_t, _x, args)
            f_startonovich = f_eval - 0.5 * jnp.einsum('kj,ijk',g_eval,dg_dx_eval)
            return f_startonovich

        if use_stratonovich:
            process_drift_used =process_drift_stratonovich


        super().__init__(n_devices,
                         n_particle_per_device,
                         initial_samples,
                         measurement_history,
                         None,
                         negative_likelihood,
                         prng_key,
                         )
        self.process_drift = process_drift_used
        self.process_diffusion = process_diffusion
        self.process_brownian_dim = process_brownian_dim
        self.dt = dt
        self.dt_meas = dt_meas
        self.resampling = resampling
        self.prng_key = prng_key
        self.sde_solver = sde_solver

        def _particle_filter_body_systematic(carry_: tuple, inputs_: tuple):
            """Particle filter body function for systematic resampling.

            Parameters
            ----------
            carry_ : tuple
                Contains (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_)
            inputs_ : tuple
                Contains (meas_, uni_, t_s_)

            Returns
            -------
            tuple
                Updated carry state containing (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_)
            tuple
                Output state containing (x_particle_resampled_, log_weights_, x_, neg_likelihood_, x_particle_)
            """
            x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            meas_, uni_, t_s_ = inputs_
            prng_key_, sub_key = jrandom.split(prng_key_)
            brownian_motion = dfx.UnsafeBrownianPath(shape=(self.n_devices, self.n_particle_per_device,
                                                            self.process_brownian_dim,), key=sub_key,
                                                     levy_area=dfx.SpaceTimeLevyArea)
            x_particle_, log_weights_, samples_sol = self._sde_propagation_and_log_weights(x_particle_resampled_, t_s_,
                                                                              meas_, brownian_motion)

            x_particle_resampled_ = systematic_or_stratified(x_particle_, log_weights_, uni_)
            neg_likelihood_ -= log_mean_exp(log_weights_)

            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.
            return (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_, x_particle_, samples_sol)

        def _particle_filter_body_stratified(carry_: tuple, inputs_: tuple):
            """Particle filter body function for stratified resampling.

            Parameters
            ----------
            carry_ : tuple
                Contains (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_)
            inputs_ : tuple
                Contains (meas_, uni_, t_s_)

            Returns
            -------
            tuple
                Updated carry state containing (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_)
            tuple
                Output state containing (x_particle_resampled_, log_weights_, x_, neg_likelihood_, x_particle_)
            """
            x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            meas_, _, t_s_ = inputs_
            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            # generate an array of uniform random number to be used for a stratified sampling
            uni_ = jrandom.uniform(subkey_, (self._n_devices * self._n_particle_per_device,))

            prng_key_, subkey_ = jrandom.split(prng_key_)
            brownian_motion = dfx.UnsafeBrownianPath(shape=(self.n_devices, self.n_particle_per_device,
                                                            self.process_brownian_dim,), key=subkey_,
                                                     levy_area=dfx.SpaceTimeLevyArea)
            x_particle_, log_weights_, samples_sol = self._sde_propagation_and_log_weights(x_particle_resampled_, t_s_,
                                                                              meas_,
                                                                              brownian_motion)
            x_particle_resampled_ = systematic_or_stratified(x_particle_, log_weights_, uni_)
            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_, x_particle_, samples_sol)

        def _particle_filter_body_multinomial(carry_: tuple, inputs_: tuple):
            """Particle filter body function for multinomial resampling.

            Parameters
            ----------
            carry_ : tuple
                Contains (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_)
            inputs_ : tuple
                Contains (meas_, uni_, t_s_)

            Returns
            -------
            tuple
                Updated carry state containing (x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_)
            tuple
                Output state containing (x_particle_resampled_, log_weights_, x_, neg_likelihood_, x_particle_)
            """
            x_particle_resampled_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_
            meas_, _, t_s_ = inputs_
            # generate random number
            prng_key_, subkey_ = jrandom.split(prng_key_)
            # generate an array of uniform random number to be used for a stratified sampling
            uni_ = jrandom.uniform(subkey_, (self._n_devices * self._n_particle_per_device + 1,))

            prng_key_, subkey_ = jrandom.split(prng_key_)
            brownian_motion = dfx.UnsafeBrownianPath(shape=(self.n_devices, self.n_particle_per_device,
                                                            self.process_brownian_dim,), key=prng_key_,
                                                     levy_area=dfx.SpaceTimeLevyArea)
            x_particle_, log_weights_, samples_sol = self._sde_propagation_and_log_weights(x_particle_resampled_, t_s_,
                                                                              meas_,
                                                                              brownian_motion)

            x_particle_resampled_ = multinomial(x_particle_, log_weights_, uni_)
            x_ = jnp.mean(x_particle_resampled_, axis=(0, 1))  # take the mean over here.
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_resampled_, log_weights_, x_, neg_likelihood_, x_particle_, samples_sol)

        def _particle_filter_body(carry_: tuple, inputs_: tuple):
            meas_, _, t_s_ = inputs_

            x_particle_, x_, log_weights_, neg_likelihood_, prng_key_ = carry_

            prng_key_, subkey_ = jrandom.split(prng_key_)
            brownian_motion = dfx.UnsafeBrownianPath(shape=(self.n_devices, self.n_particle_per_device,
                                                            self.process_brownian_dim,), key=subkey_,
                                                     levy_area=dfx.SpaceTimeLevyArea)
            x_particle_, log_weights_, samples_sol = self._sde_propagation_and_log_weights(x_particle_, t_s_,
                                                                              meas_,
                                                                              brownian_motion)
            # No resampling
            x_ = jnp.sum(x_particle_ * jnp.exp(log_weights_[:, :, jnp.newaxis]), axis=(0, 1))
            # take the mean over here.

            # normalize log weight here
            log_weights_ = normalize_log_weights(log_weights_)
            neg_likelihood_ -= log_mean_exp(log_weights_)
            return (x_particle_, x_, log_weights_, neg_likelihood_, prng_key_), (
                x_particle_, log_weights_, x_, neg_likelihood_, x_particle_, samples_sol)

        if self.resampling.lower() == 'systematic':
            self.particle_filter_body = _particle_filter_body_systematic
        elif self.resampling.lower() == 'stratified':
            self.particle_filter_body = _particle_filter_body_stratified
        elif self.resampling.lower() == 'multinomial':
            self.particle_filter_body = _particle_filter_body_multinomial
        else:
            self.particle_filter_body = _particle_filter_body

    @filter_jit
    def _sde_propagation_and_log_weights(self, x_particle_initial_: Array,
                                         t_s: float, meas_: Array,
                                         brownian_motion_: dfx.UnsafeBrownianPath):
        """Propagates particles through SDE and computes log weights.

        This method takes the initial particle states and propagates them forward in time
        according to the specified stochastic differential equation (SDE). It then
        computes log weights based on the likelihood of the measurements.

        Parameters
        ----------
        x_particle_initial_ : np.ndarray
            Initial particle states
        t_s : float
            Start time for propagation
        meas_ : np.ndarray
            Measurement data
        brownian_motion_ : dfx.UnsafeBrownianPath
            Brownian motion path for SDE simulation

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - Propagated particle states after SDE evolution
            - Log weights computed from measurement likelihood
        """
        # propagate according to the sde
        sde_terms = dfx.MultiTerm(dfx.ODETerm(self.process_drift),
                                  VectorizedControlTerm(self.process_diffusion,
                                                        brownian_motion_))

        samples_sol = dfx.diffeqsolve(sde_terms,
                                      self.sde_solver,
                                      t_s,  # starting time
                                      t_s + self.dt_meas,  # end time
                                      dt0=self.dt,  # sde solver delta t
                                      y0=x_particle_initial_,
                                      saveat=dfx.SaveAt(t1=True),  #only saves at the end
                                      adjoint=dfx.DirectAdjoint(),
                                      max_steps=int(1e12))

        x_particle_after_ = samples_sol.ys[-1]

        # compute the log_weights
        log_weights_ = -self.neg_log_likelihood(x_particle_after_, meas_)

        return x_particle_after_, log_weights_, samples_sol

    def run(self) -> tuple[Array, Array, Array, Array, Array, Array, Array,
    Any, timedelta]:
        # no need to be normalized
        log_weigths_init = jnp.zeros((self.n_devices, self.n_particle_per_device))
        neg_likelihood_init = 0
        carry = (self.initial_samples, self.mean_init,
                 log_weigths_init, neg_likelihood_init,
                 self.prng_key)

        # this part differs from the parent class
        n_meas = self.measurement_history.shape[0]  # how many data points in time series
        measurement_time = self.dt_meas * (jnp.arange(n_meas))
        inputs = (self.measurement_history, self.uniforms,
                  measurement_time - self.dt_meas)

        @filter_jit
        def to_be_compiled(_carry_init, _inputs):
            carry_end, output = lax.scan(self.particle_filter_body,
                                         _carry_init,
                                         _inputs)
            return carry_end, output

        # compiled_run is needed to check the analytics like FLOPS etc
        compiled_run = to_be_compiled.lower(carry, inputs).compile()
        start_time = datetime.now()
        (x_particle_end, state_end,
         log_weights_end, neg_log_likelihood_end,
         a_prng_key), res = compiled_run(carry, inputs)


        x_particle_end.block_until_ready()
        end_time = datetime.now()
        execution_time = end_time - start_time

        (x_particle_resampled_history, log_weights_history,
         estimated_state_history, neg_likelihood_history,
         x_particle_history, samples_sol_history) = res
        return neg_log_likelihood_end, state_end, x_particle_resampled_history, neg_likelihood_history, \
            log_weights_history, estimated_state_history, x_particle_history, compiled_run, execution_time, samples_sol_history


def create_particle_drift_for_cost_analysis(
    process_drift: Callable[[float, Array, PyTree], Array],
) -> Callable:
    """
    Create a standalone JIT-compiled particle filter drift function for cost analysis.

    For particle filter, we measure the drift term separately since the SDE uses MultiTerm.
    The diffusion term cost is typically dominated by the drift for large ensembles.

    Parameters
    ----------
    process_drift : Callable[[float, Array, PyTree], Array]
        The drift function f(t, x, args) of the SDE.

    Returns
    -------
    Callable
        JIT-compiled drift function with signature (t, samples, args) -> d_samples
    """
    @filter_jit
    def drift_derivative(t, samples, args=None):
        return process_drift(t, samples, args)

    return drift_derivative


def create_particle_update_for_cost_analysis(
    negative_log_likelihood: Callable[[Array, Array], Array],
) -> Callable:
    """
    Create a standalone JIT-compiled particle filter weight update function for cost analysis.

    This measures the cost of computing log weights from the negative log likelihood.
    Note: Resampling cost is not included as it's typically O(N) and much smaller than
    the likelihood evaluation.

    Parameters
    ----------
    negative_log_likelihood : Callable[[Array, Array], Array]
        The negative log likelihood function -log p(y|x).

    Returns
    -------
    Callable
        JIT-compiled update function with signature (samples, meas) -> log_weights
    """
    @filter_jit
    def particle_update(samples: Array, meas: Array):
        log_weights = -negative_log_likelihood(samples, meas)
        return log_weights

    return particle_update
