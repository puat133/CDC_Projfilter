# Define your SimulationConfig data class (if you haven't already)
import chex
import pathlib
import jax.numpy as jnp
import diffrax as dfx
from jaxtyping import Array
from exponential_family.n_d_ef import NDExponentialFamily
from enum import Enum

Simulation_Case = Enum('Simulation_Case', [('PROJECTION_FILTER_ONLY', 1),
                                           ('BENCHMARK_ONLY', 2),
                                           ('FULL', 3)])

@chex.dataclass
class SimulationConfig:
    dt: float = 1e-5
    t_f: float = 1e0
    var_scale: float = 1.0
    mean_scale: float = 1.0
    model_scale: float = 1.0
    n_meas: int = 5
    seed: int = 2
    n_point_per_axis: int = 101
    n_std: int = 6
    use_linear_meas: bool = True
    use_multi_core_cpu: bool = True
    save: bool = True
    save_particle_samples: bool = False
    gauss_mixture_clusters: int = 25
    gauss_mixture_samples: int = int(1e5)
    em_max_iterations: int = 10
    n_sw_projection: int = 100
    n_ef_samples: int = 10000
    sim_path: pathlib.Path = pathlib.Path("")
    sim_case: Simulation_Case = Simulation_Case.PROJECTION_FILTER_ONLY


@chex.dataclass
class ParticleFilterConfig:
    n_particle_per_device: int = int(1e4)
    resampling: str = "systematic"
    use_stratonovich: bool = True
    sde_solver: dfx.AbstractSolver = dfx.EulerHeun()


@chex.dataclass
class EnKFConfig:
    n_particle_per_device: int = int(1e4)

# TODO: Define QuadratureConfig data class
# @chex.dataclass
# class QuadratureConfig:
#     type: str  = "SPG" # Either SPQ, Hermite, or QMC
#     level: int = 5     # sparse grid level
#     nodes_number: int =1000 # only matter if the type is QMC
#     rule: str  = "gauss-patterson" # only matter if SPG type is used


@chex.dataclass
class ProjectionFilterConfig:
    theta_ell_args: tuple[Array, Array]
    theta_init: Array
    params_init: tuple[Array, Array, float]
    theta_indices_for_bijection_params: tuple[ Array, Array]
    constant_step_size: bool = False
    rtol: float = 1e-6
    atol: float = 1e-9
    dt: float = 1e-5
    dt_prep: float = 1e-5
    mmt_iter: int = 6
    par_scale: float = 1.0
    s_level: int = 8
    max_order_monomials: int = 4
    ode_max_steps: int = int(1e4)
    spg_rule: str = "gauss-patterson"
    direct_calculation: bool = False
    save_fokker_planck_thetas: bool = False
    ode_solver: dfx.AbstractSolver = dfx.Tsit5()
    ode_solver_prep: dfx.AbstractSolver = dfx.Tsit5()
    alpha_prep: float = 0.0
    beta_prep: float = 4.0
    t_f_prep: float = 1e2
    learn_rate_init_prep: float = 1.0
    p_pid: float = 0.0
    i_pid: float = 1.0
    d_pid: float = 0.
    n_devices: int = 1
    n_particle_per_device: int = 1
    fisher_regularizer_initial_lambda: float = float(jnp.power(2.0,-32)),
    fisher_regularizer_lambda_factor: int = 5,
    fisher_regularizer_max_attempts: int = 15
    max_d_theta_dt_norm: float = jnp.inf
    min_fisher_ev: float = 0