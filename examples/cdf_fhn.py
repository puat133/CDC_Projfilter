import logging
import multiprocessing
import os
import pathlib

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(multiprocessing.cpu_count())

import numpy as onp
import argparse
from datetime import datetime
from utils.boolean_parser import add_boolean_argument

import jax

jax.config.update("jax_enable_x64", True)  # this must be enabled, or the particle filter simulation will be wrong
jax.config.update("jax_debug_nans", True)
DEBUG_MODE = False
if DEBUG_MODE:
    jax.config.update("jax_disable_jit", True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_traceback_filtering", "off")

import jax.numpy as jnp
import sympy as sp
import jax.random as jrandom
from equinox import filter_jit
from functools import partial
from utils.density_manipulations import calculate_gaussian_natural_parameters
from cd_filtering.cd_proj_conjugate import get_T_matrices
import ott.tools.gaussian_mixture.gaussian_mixture as gm
import diffrax as dfx
from symbolic.n_d import SDE
from symbolic.n_d import mix_monomials_up_to_order
from symbolic.sympy_to_jax import sympy_matrix_to_jax
from simulation.configs import (
    SimulationConfig,
    EnKFConfig,
    ParticleFilterConfig,
    ProjectionFilterConfig,
    HGMFConfig,
    Simulation_Case,
)
from simulation.dynamical_system import DynamicalSystem
from simulation.simulation import (simulate_fully, simulate_projection_filter_only,
                                   simulate_benchmark, simulate_hgmf_only)

# FitzHugh-Nagumo default parameters
DEFAULT_A = 0.7
DEFAULT_B = 0.8
DEFAULT_TAU = 12.5
DEFAULT_I1 = 0.25
DEFAULT_I2 = 0.5
DEFAULT_EPS12 = 0.1
DEFAULT_EPS21 = 0.1
DEFAULT_SIGMA_W = 1.0
DEFAULT_SIGMA_V = 2.0
DEFAULT_DT = 1e-2
DEFAULT_FINAL_TIME = 1e0
DEFAULT_VAR_SCALE = 1.0
DEFAULT_PAR_SCALE = 1.0
DEFAULT_MEAN_SCALE = 1.0
DEFAULT_N_MEAS = 4
DEFAULT_MMT_ITER = 1
DEFAULT_N_PARTICLE = 1000000
DEFAULT_SEED = 1
DEFAULT_S_LEVEL = 6
DEFAULT_N_POINT = 150
DEFAULT_MAX_ORDER = 4
DEFAULT_N_PGM_SMPLS = (10*186345)  # equivalent to 10 times G-Kronrod level 6 for 4D
DEFAULT_PGM_CLSTR = 50
DEFAULT_ODE_MAX_STEPS = int(1e2)
DEFAULT_ODE_SOLVER = dfx.Tsit5()
DEFAULT_PREP_ODE_SOLVER = dfx.Tsit5()
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = DEFAULT_RTOL * 1e-3
DEFAULT_REGULARIZER_FACTOR = 0e0
DEFAULT_REGULARIZER_SCALE = 7
DEFAULT_REGULARIZER_ORDER_INCREASE = 8
DEFAULT_FISHER_REGULARIZER_SCALE = 1e-8
DEFAULT_RESAMPLING = "systematic"
DEFAULT_SPG_RULE = "gauss-kronrod"
DEFAULT_QMC_NODES_NUM = int(1e5)
DEFAULT_MLTCORE = True
DEFAULT_SAVE = True
DEFAULT_PROGRESS_BAR = False
DEFAULT_CONSTANT_STEP_SIZE = False
DEFAULT_MAX_D_THETA_DT_NORM = 1e1
DEFAULT_MIN_FISHER_EV = 0

if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser()
    # FHN-specific parameters
    parser.add_argument("--a", default=DEFAULT_A, type=float, help="FHN parameter a")
    parser.add_argument("--b", default=DEFAULT_B, type=float, help="FHN parameter b")
    parser.add_argument("--tau", default=DEFAULT_TAU, type=float, help="FHN time scale separation tau")
    parser.add_argument("--I1", default=DEFAULT_I1, type=float, help="External input to oscillator 1")
    parser.add_argument("--I2", default=DEFAULT_I2, type=float, help="External input to oscillator 2")
    parser.add_argument("--eps12", default=DEFAULT_EPS12, type=float, help="Coupling from osc 2 to osc 1")
    parser.add_argument("--eps21", default=DEFAULT_EPS21, type=float, help="Coupling from osc 1 to osc 2")

    # Noise parameters
    parser.add_argument(
        "--sigmaw", default=DEFAULT_SIGMA_W, type=float, help="scaling factor for the dynamic Wiener process"
    )
    parser.add_argument(
        "--sigmav", default=DEFAULT_SIGMA_V, type=float, help="scaling factor for the measurement Wiener process"
    )

    # Time and simulation parameters
    parser.add_argument("--dt", default=DEFAULT_DT, type=float, help="sampling time")
    parser.add_argument("--tf", default=DEFAULT_FINAL_TIME, type=float, help="final simulation time")
    parser.add_argument(
        "--varscale", default=DEFAULT_VAR_SCALE, type=float, help="Scalar multiplier to identity for initial variance"
    )
    parser.add_argument("--scale", default=DEFAULT_PAR_SCALE, type=float, help="bijection parameter scale factor")
    parser.add_argument("--mean_scale", default=DEFAULT_MEAN_SCALE, type=float, help="mean scale factor")
    parser.add_argument('--max_d_theta_dt_norm', default=DEFAULT_MAX_D_THETA_DT_NORM, type=float, help='maximum allowed d_theta_dt_norm')
    parser.add_argument('--min_fisher_ev', default=DEFAULT_MIN_FISHER_EV, type=float, help='minimum eigenvalue for Fisher truncation')
    parser.add_argument("--n_meas", default=DEFAULT_N_MEAS, type=int, help="how many measurements are taken")
    parser.add_argument(
        "--mmt_iter", default=DEFAULT_MMT_ITER, type=int, help="moment_matching_iteration_number"
    )
    parser.add_argument(
        "--n_particle",
        default=DEFAULT_N_PARTICLE,
        type=int,
        help="number of particle for particle filter for each processing device",
    )
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int, help="prngkey for jax.random")
    parser.add_argument("--s_level", default=DEFAULT_S_LEVEL, type=int, help="sparse grid integration level")
    parser.add_argument("--n_point", default=DEFAULT_N_POINT, type=int, help="number of particles per dimension")
    parser.add_argument(
        "--max_order_monomials",
        default=DEFAULT_MAX_ORDER,
        type=int,
        help="maximum total order of monomials in the natural statistics",
    )
    parser.add_argument("--pgm_samples", default=DEFAULT_N_PGM_SMPLS, type=int, help="Number of pgm samples")
    parser.add_argument("--pgm_clusters", default=DEFAULT_PGM_CLSTR, type=int, help="Number of pgm clusters")
    parser.add_argument("--hgmf_sp_order", default=5, type=int,
                        help="HGMF prediction sigma-point order (Gauss-Hermite)")
    parser.add_argument("--hgmf_eps", default=1e-3, type=float,
                        help="HGMF log-homotopy epsilon (Corollary 5)")
    parser.add_argument("--hgmf_rtol_hom", default=1e-3, type=float,
                        help="HGMF homotopic-correction PID rtol")
    parser.add_argument("--hgmf_atol_hom", default=1e-6, type=float,
                        help="HGMF homotopic-correction PID atol")
    parser.add_argument("--hgmf_dt_hom", default=1e-2, type=float,
                        help="HGMF homotopic-correction initial step size")
    add_boolean_argument(parser, "hgmf_log_homotopy", default=True,
                         messages="whether to integrate the HGMF correction in tau = log(s + eps) instead of s.")
    parser.add_argument(
        "--sim_case", default=1, type=int,
        help="1 - Projection Filter Only, 2 - Benchmark Only, 3 - Full, 4 - HGMF Only"
    )

    parser.add_argument(
        "--resampling", default=DEFAULT_RESAMPLING, type=str, help="resampling method for particle filter"
    )
    parser.add_argument("--srule", default=DEFAULT_SPG_RULE, type=str, help="sparse grid integration rule")
    add_boolean_argument(
        parser,
        "multicore",
        default=DEFAULT_MLTCORE,
        messages="whether to consider each core as separate processing unit",
    )
    add_boolean_argument(parser, "save", default=DEFAULT_SAVE, messages="whether to save the variables to a hdf file.")
    add_boolean_argument(parser, "save_samples", default=False, messages="whether to also save particle and enkf samples.")
    add_boolean_argument(
        parser, "progress_bar", default=DEFAULT_PROGRESS_BAR, messages="whether to show the progress bar."
    )
    add_boolean_argument(
        parser,
        "constant_step_size",
        default=DEFAULT_CONSTANT_STEP_SIZE,
        messages="whether to use a constant step size in the ode solver.",
    )

    args = parser.parse_args()

    # set simulation path
    simulation_time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    simulation_id = "FHN_SEED_{}_".format(args.seed) + simulation_time_string
    sim_case = Simulation_Case(args.sim_case)
    if sim_case == Simulation_Case.HGMF_ONLY:
        sim_path = pathlib.Path(f'./simulation_result/FHN-{args.max_order_monomials}-{str(sim_case)}-TF_{args.tf}-NMEAS_{args.n_meas}/{simulation_id}')
    else:
        sim_path = pathlib.Path(f'./simulation_result/FHN-{args.max_order_monomials}-{str(sim_case)}-F_MIN_EV_{args.min_fisher_ev}-MAX_DTHETA_DT_{args.max_d_theta_dt_norm}-TF_{args.tf}-NMEAS_{args.n_meas}/{simulation_id}')

    if not sim_path.exists():
        sim_path.mkdir(parents=True, exist_ok=True)

    if not sim_path.exists():
        sim_path.mkdir(parents=True, exist_ok=True)

    # set configs
    sim_config = SimulationConfig(
        dt=args.dt,
        t_f=args.tf,
        mean_scale=args.mean_scale,
        n_meas=args.n_meas,
        seed=args.seed,
        n_point_per_axis=args.n_point,
        n_std=6,
        use_linear_meas=True,
        use_multi_core_cpu=args.multicore,
        save=args.save,
        gauss_mixture_clusters=args.pgm_clusters,
        gauss_mixture_samples=args.pgm_samples,
        em_max_iterations=10,
        sim_path=sim_path,
        sim_case=sim_case,
        save_particle_samples=args.save_samples,
    )

    par_config = ParticleFilterConfig(
        n_particle_per_device=args.n_particle,
        resampling=args.resampling,
        use_stratonovich=True,
        sde_solver=dfx.EulerHeun(),
    )
    enkf_config = EnKFConfig(
        n_particle_per_device=args.n_particle,
    )

    hgmf_config = HGMFConfig(sp_order=args.hgmf_sp_order,
                             use_log_homotopy=args.hgmf_log_homotopy,
                             log_homotopy_eps=args.hgmf_eps,
                             rtol_pred=DEFAULT_RTOL,
                             atol_pred=DEFAULT_ATOL,
                             rtol_hom=args.hgmf_rtol_hom,
                             atol_hom=args.hgmf_atol_hom,
                             dt_pred=args.dt,
                             dt_hom=args.hgmf_dt_hom,
                             constant_step_size=args.constant_step_size)

    # set FHN parameters
    a_fhn = args.a
    b_fhn = args.b
    tau_fhn = args.tau
    I1_fhn = args.I1
    I2_fhn = args.I2
    eps12_fhn = args.eps12
    eps21_fhn = args.eps21
    sigma_w_fhn = args.sigmaw
    sigma_v_fhn = args.sigmav
    max_order_monomials = args.max_order_monomials

    # FHN is 4D system (coupled oscillators)
    dim_states = 4

    # Measurement dimension: monomials of degree max_order/2
    # For max_order=4, degree=2 monomials in 4D: x1^2, x1*x2, x1*x3, x1*x4, x2^2, x2*x3, x2*x4, x3^2, x3*x4, x4^2
    # This gives C(4+2-1, 2) - C(4+1-1, 1) = 10 monomials of exactly degree 2

    dynamic_parameters = {
        "a": a_fhn,
        "b": b_fhn,
        "tau": tau_fhn,
        "I1": I1_fhn,
        "I2": I2_fhn,
        "eps12": eps12_fhn,
        "eps21": eps21_fhn,
        "sigma_w": sigma_w_fhn,
        "sigma_v": sigma_v_fhn,
    }

    if sim_config.use_multi_core_cpu:
        n_devices = jax.local_device_count()
    else:
        n_devices = 1

    # set natural statistics (4D symbolic variables)
    x_sp, dw_sp, dv_sp = sp.symbols(("x1:5", "dw1:5", "dv1:5"))
    t_sp = sp.symbols("t")

    # FHN Dynamics (coupled system):
    # dx1 = [x1 - x1^3/3 - x2 + I1 + eps12*(x3 - x1)] dt + sigma_w * dW1
    # dx2 = [(x1 + a - b*x2) / tau] dt + sigma_w * dW2
    # dx3 = [x3 - x3^3/3 - x4 + I2 + eps21*(x1 - x3)] dt + sigma_w * dW3
    # dx4 = [(x3 + a - b*x4) / tau] dt + sigma_w * dW4
    f1 = x_sp[0] - x_sp[0] ** 3 / 3 - x_sp[1] + I1_fhn + eps12_fhn * (x_sp[2] - x_sp[0])
    f2 = (x_sp[0] + a_fhn - b_fhn * x_sp[1]) / tau_fhn
    f3 = x_sp[2] - x_sp[2] ** 3 / 3 - x_sp[3] + I2_fhn + eps21_fhn * (x_sp[0] - x_sp[2])
    f4 = (x_sp[2] + a_fhn - b_fhn * x_sp[3]) / tau_fhn

    f = sp.Matrix([f1, f2, f3, f4])
    g = sp.diag(sigma_w_fhn, sigma_w_fhn, sigma_w_fhn, sigma_w_fhn)
    dynamic_sde = SDE(f, g, t_sp, x_sp, dw_sp)

    # Natural statistics: all monomials up to max_order
    natural_statistics_symbolic_original = mix_monomials_up_to_order(x_sp, max_order_monomials)

    # Use polynomial measurement functions: monomials of degree max_order/2
    ch_symbolic_list = []
    for a_stat in natural_statistics_symbolic_original:
        if a_stat.as_poly(x_sp).total_degree() == 1:
            ch_symbolic_list.append(a_stat)
    ch_symbolic = sp.Matrix(ch_symbolic_list)
    ch_symbolic_jax, _ = sympy_matrix_to_jax(ch_symbolic, x_sp)
    
    # add an extra natural statistics for numerical stability
    stat_extra = 0
    for x_sp_i in x_sp:
        stat_extra += x_sp_i**(max_order_monomials + 2)

    natural_statistics_symbolic = [stat[0] for stat in natural_statistics_symbolic_original.tolist()]
    natural_statistics_symbolic.append(stat_extra)
    natural_statistics_symbolic = sp.Matrix(natural_statistics_symbolic)

    # Measurement covariance
    meas_dim = len(ch_symbolic)
    R = sigma_v_fhn**2 * jnp.eye(meas_dim)
    R_inv = jnp.linalg.inv(R)

    @filter_jit
    @partial(jnp.vectorize, signature="(n)->(n)", excluded=(0, 2))
    def drift(_t: float, _x: jnp.ndarray, args) -> jnp.ndarray:
        """
        the drift part of the FHN dynamic.
        """
        x1, x2, x3, x4 = _x[0], _x[1], _x[2], _x[3]
        dx1 = x1 - x1**3 / 3 - x2 + I1_fhn + eps12_fhn * (x3 - x1)
        dx2 = (x1 + a_fhn - b_fhn * x2) / tau_fhn
        dx3 = x3 - x3**3 / 3 - x4 + I2_fhn + eps21_fhn * (x1 - x3)
        dx4 = (x3 + a_fhn - b_fhn * x4) / tau_fhn
        return jnp.array([dx1, dx2, dx3, dx4])

    @filter_jit
    @partial(jnp.vectorize, signature="(n)->(n,m)", excluded=(0, 2))
    def diffusion(_t: float, _x: jnp.ndarray, args) -> jnp.ndarray:
        """
        the diffusive part of the FHN dynamic (diagonal noise).
        """
        return jnp.diag(jnp.full((4,), sigma_w_fhn))

    @filter_jit
    @partial(jnp.vectorize, signature="(d)->(m)")
    def meas_fun(_x: jnp.ndarray) -> jnp.ndarray:
        """
        the measurement function: monomials of degree max_order/2.
        For max_order=4, this gives degree-2 monomials.
        """
        return ch_symbolic_jax(_x)

    @filter_jit
    @partial(jnp.vectorize, signature="(d)->()", excluded=(1,))
    def ell_y(_x: jnp.ndarray, _meas: jnp.ndarray) -> jnp.ndarray:
        _h_eval = meas_fun(_x)
        return 0.5 * ((_meas - _h_eval).T @ R_inv @ (_meas - _h_eval))

    # prepare dynamical system
    dynamic = DynamicalSystem(
        dim_states=dim_states,
        dim_output=meas_dim,
        dim_process_brownian=dim_states,  # 4D process noise
        dim_meas_brownian=meas_dim,
        sde=dynamic_sde,
        drift=drift,
        diffusion=diffusion,
        measurement=meas_fun,
        negative_log_likelihood=ell_y,
        parameters=dynamic_parameters
    )

    # preparing ProjectionFilterConfig
    T1, T2 = get_T_matrices(ch_symbolic, natural_statistics_symbolic)
    H = jnp.eye(T1.shape[0])
    theta_ell_args = T1.T @ H @ R_inv, 0.5 * T2.T @ (H @ R_inv @ H.T).flatten()

    # theta init
    # create mixture Gaussian initial samples (4D)
    mean_1 = sim_config.mean_scale * jnp.ones(dim_states)
    mean_2 = -mean_1
    means_init = jnp.array([mean_1, mean_2])
    cov_init = (sim_config.model_scale**2) * sim_config.var_scale * jnp.eye(dim_states)
    covs_init = jnp.array([cov_init, cov_init])
    gmm = gm.GaussianMixture.from_mean_cov_component_weights(means_init, covs_init, jnp.array([0.5, 0.5]))

    prng_key = jrandom.PRNGKey(sim_config.seed)
    prng_key, sub_key = jrandom.split(prng_key)
    init_samples = gmm.sample(sub_key, n_devices * par_config.n_particle_per_device)
    init_samples = init_samples.reshape((n_devices, par_config.n_particle_per_device, dim_states))
    init_mean = 0.5 * (mean_1 + mean_2)

    var_init_inv = jnp.linalg.solve(cov_init, jnp.eye(dim_states))
    gaussian_initial_condition_original, theta_indices_for_bijection_params = calculate_gaussian_natural_parameters(
        init_mean,
        cov_init,
        natural_statistics_symbolic,
        dynamic.sde.variables,
    )

    # No additional statistics for FHN (unlike VDP which has trig terms)
    gaussian_initial_condition = gaussian_initial_condition_original

    # cholesky version is used
    intial_bijection_parameters_gaussian = (init_mean, jnp.linalg.cholesky(cov_init), args.scale)

    rtol = DEFAULT_RTOL
    atol = DEFAULT_ATOL
    proj_config = ProjectionFilterConfig(
        theta_ell_args=theta_ell_args,
        theta_init=gaussian_initial_condition,
        params_init=intial_bijection_parameters_gaussian,
        theta_indices_for_bijection_params=theta_indices_for_bijection_params,
        constant_step_size=args.constant_step_size,
        rtol=rtol,
        atol=atol,
        dt=args.dt,
        dt_prep=args.dt,
        mmt_iter=args.mmt_iter,
        par_scale=args.scale,
        s_level=args.s_level,
        max_order_monomials=args.max_order_monomials,
        ode_max_steps=DEFAULT_ODE_MAX_STEPS,
        spg_rule=args.srule,
        direct_calculation=True,
        save_fokker_planck_thetas=True,
        ode_solver=DEFAULT_ODE_SOLVER,
        ode_solver_prep=DEFAULT_PREP_ODE_SOLVER,
        fisher_regularizer_initial_lambda = 1e-30,
        fisher_regularizer_lambda_factor = 10,
        fisher_regularizer_max_attempts = 30,
        min_fisher_ev=args.min_fisher_ev,
        max_d_theta_dt_norm=args.max_d_theta_dt_norm
    )

    # the initial state is taken randomly from the initial sample
    prng_key, sub_key = jrandom.split(prng_key)
    random_index = jrandom.randint(sub_key, (1,), 0, (init_samples.shape[0] * init_samples.shape[1] - 1))
    init_state = init_samples.reshape(init_samples.shape[0] * init_samples.shape[1], -1)[0].squeeze()

    # Prepare the Gauss Mixture things
    means_gm_init = jnp.array(
        [
            mean_1,
            mean_2,
            *[
                onp.random.randn(
                    dim_states,
                )
                for i in range(sim_config.gauss_mixture_clusters - 2)
            ],
        ]
    )
    covs_gm_init = jnp.array([cov_init for i in range(means_gm_init.shape[0])])
    small_probability = 1e-2 / sim_config.gauss_mixture_clusters
    weights_gm_init = jnp.array([0.5, 0.5, *[small_probability for i in range(sim_config.gauss_mixture_clusters - 2)]])
    weights_gm_init = weights_gm_init / jnp.sum(weights_gm_init)

    # prepare logger
    logger = logging.getLogger("FHN")
    log_file = sim_path / "simulation.log"
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    # Create a formatter that includes the timestamp
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # Set the formatter for the handler
    handler.setFormatter(formatter)

    progress_bar_type = ""
    if args.progress_bar:
        progress_bar_type = "text"

    if sim_config.sim_case == Simulation_Case.PROJECTION_FILTER_ONLY:
        simulation_result = simulate_projection_filter_only(
            dynamic,
            sim_config,
            proj_config,
            init_state,
            init_samples,
            natural_statistics_symbolic,
            logger,
            progress_bar_type
        )
    elif sim_config.sim_case == Simulation_Case.BENCHMARK_ONLY:
        simulation_result = simulate_benchmark(
            dynamic,
            sim_config,
            proj_config,
            par_config,
            enkf_config,
            init_state,
            init_samples,
            natural_statistics_symbolic,
            means_gm_init,
            covs_gm_init,
            weights_gm_init,
            logger,
            hgmf_config=hgmf_config,
        )
    elif sim_config.sim_case == Simulation_Case.FULL:
        simulation_result = simulate_fully(
            dynamic,
            sim_config,
            proj_config,
            par_config,
            enkf_config,
            init_state,
            init_samples,
            natural_statistics_symbolic,
            means_gm_init,
            covs_gm_init,
            weights_gm_init,
            logger,
            progress_bar_type,
            hgmf_config=hgmf_config,
        )
    elif sim_config.sim_case == Simulation_Case.HGMF_ONLY:
        simulation_result = simulate_hgmf_only(
            dynamic,
            sim_config,
            init_state,
            init_samples,
            means_gm_init,
            covs_gm_init,
            weights_gm_init,
            logger,
            hgmf_config=hgmf_config,
        )
    else:
        print("Unknown simulation case")
