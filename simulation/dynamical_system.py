import jax.numpy as jnp
import jax.random as jrandom
from symbolic.n_d import SDE
from typing import Callable

import diffrax as dfx
from utils.diffrax import VectorizedControlTerm
from diffrax import ControlTerm
from jaxtyping import Array, Float, PyTree

class DynamicalSystem:
    dim_states: int
    dim_output: int
    dim_process_brownian: int
    dim_meas_brownian: int
    sde: SDE
    drift : Callable[[Float, Array, PyTree], Array]
    diffusion : Callable[[Float, Array, PyTree], Array]
    measurement : Callable[[Array], Array]
    negative_log_likelihood: Callable[[Array, Array], Array]
    parameters : dict[str, PyTree]

    def __init__(self,
        dim_states: int,
        dim_output: int,
        dim_process_brownian: int,
        dim_meas_brownian: int,
        sde: SDE,
        drift: Callable[[Float, Array, PyTree], Array],
        diffusion: Callable[[Float, Array, PyTree], Array],
        measurement: Callable[[Array], Array],
        negative_log_likelihood: Callable[[Array, Array], Array],
        parameters):

        self.dim_states = dim_states
        self.dim_output = dim_output
        self.dim_process_brownian = dim_process_brownian
        self.dim_meas_brownian = dim_meas_brownian
        self.sde = sde
        self.drift = drift
        self.diffusion = diffusion
        self.measurement = measurement
        self.negative_log_likelihood = negative_log_likelihood
        self.parameters = parameters


    def generate_path_and_measurement(self,
                                      init_state: Array,
                                      t_span:Array,
                                      dt:Float,
                                      prng_key:Array,
                                      solver: dfx.AbstractSolver=dfx.Euler())->tuple[Array,
                                                                   Array,
                                                                   Array,
                                                                   Float]:

        t_s = t_span[0]
        t_f = t_span[-1]
        n_meas = t_span.shape[0]-1

        brownian_motion = dfx.VirtualBrownianTree(t_s, t_f, tol=dt, shape=(self.dim_process_brownian,), key=prng_key,
                                                  levy_area=dfx.SpaceTimeLevyArea,
                                                  )
        terms = dfx.MultiTerm(dfx.ODETerm(self.drift),
                              VectorizedControlTerm(self.diffusion, brownian_motion))
        saveat = dfx.SaveAt(ts=t_span)

        sol = dfx.diffeqsolve(terms, solver, t_s, t_f, dt0=dt, y0=init_state, saveat=saveat, max_steps=int(1e12))
        x_integrated = sol.ys

        time_sample = float((t_f - t_s) / n_meas)  # meas_skip * dt
        t_meas = t_span  # (jnp.arange(nt_total // meas_skip) + 1) * time_sample

        prng_key, subkey = jrandom.split(prng_key)
        meas_noise = self.parameters["sigma_v"] * jrandom.normal(subkey, (x_integrated.shape[0]-1, self.dim_output))
        measurement_record = self.measurement(x_integrated[1:]) + meas_noise

        return x_integrated, measurement_record, t_meas, time_sample
