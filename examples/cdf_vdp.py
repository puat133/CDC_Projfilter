import logging
import multiprocessing
import os
import pathlib

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())

import numpy as onp
import argparse
from datetime import datetime
from utils.boolean_parser import add_boolean_argument

import jax
jax.config.update("jax_enable_x64", True)  # this must be enabled, or the particle filter simulation will be wrong
DEBUG_MODE = False
if DEBUG_MODE:
    jax.config.update('jax_disable_jit', True)
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_traceback_filtering", "off")

import jax.numpy as jnp
import sympy as sp
import jax.random as jrandom
from equinox import filter_jit
from functools import partial
from utils.density_manipulations import calculate_gaussian_natural_parameters
from cd_filtering.cd_proj_conjugate import (get_T_matrices)
import ott.tools.gaussian_mixture.gaussian_mixture as gm
import diffrax as dfx
from symbolic.n_d import SDE
from symbolic.n_d import mix_monomials_up_to_order

from simulation.configs import (SimulationConfig, EnKFConfig, ParticleFilterConfig,
                                ProjectionFilterConfig, Simulation_Case)
from simulation.dynamical_system import DynamicalSystem
from simulation.simulation import simulate_fully, simulate_projection_filter_only, simulate_benchmark

# Enable debugging
# jax.debug_infs()
# jax.debug_nans()



DEFAULT_KAPPA = 0.0
DEFAULT_MU = 0.5
DEFAULT_SIGMA_W = 1.0
DEFAULT_SIGMA_V = 2.0
DEFAULT_DT = 5e-2
DEFAULT_FINAL_TIME = 1e0
DEFAULT_VAR_SCALE = 1.0
DEFAULT_PAR_SCALE = 1.0 # 1.2 Default is 1.2 with gauss-patterson
DEFAULT_MEAN_SCALE = 1.0
DEFAULT_N_MEAS = 4
DEFAULT_MMT_ITER = 1
DEFAULT_N_PARTICLE = 1000000
DEFAULT_SEED = 1
DEFAULT_S_LEVEL = 8
DEFAULT_N_POINT = 150
DEFAULT_MAX_ORDER = 4
DEFAULT_N_PGM_SMPLS = 208330 #this is 10x used quadrature nodes for projection fitler. #int(1e5) equal to projection filter quadrature nodes
DEFAULT_PGM_CLSTR = 50
DEFAULT_ODE_MAX_STEPS = int(1e3)
DEFAULT_ODE_SOLVER = dfx.Tsit5()
DEFAULT_PREP_ODE_SOLVER = dfx.Tsit5()
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = DEFAULT_RTOL * 1e-3
DEFAULT_REGULARIZER_FACTOR = 0e-8 # default stabilization is 1e-8
DEFAULT_REGULARIZER_SCALE = 10 # default is 2, related to ((x-mu)/scale)
DEFAULT_REGULARIZER_ORDER_INCREASE = 2 # default is 6
DEFAULT_FISHER_REGULARIZER_SCALE = 1e-8 # Default is 1e1
DEFAULT_RESAMPLING = "systematic"
DEFAULT_SPG_RULE = "gauss-kronrod"
DEFAULT_QMC_NODES_NUM = int(1e5)
DEFAULT_MLTCORE = True
DEFAULT_SAVE = True
DEFAULT_PROGRESS_BAR = False
DEFAULT_CONSTANT_STEP_SIZE = False
DEFAULT_MAX_D_THETA_DT_NORM = jnp.inf
DEFAULT_MIN_FISHER_EV = 0

if __name__ == '__main__':

    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--kappa', default=DEFAULT_KAPPA, type=float, help='kappa for the modified vdp dynamic')
    parser.add_argument('--mu', default=DEFAULT_MU, type=float, help='mu for the modified vdp dynamic')
    parser.add_argument('--sigmaw', default=DEFAULT_SIGMA_W, type=float,
                        help='scaling factor for the dyanmic Wiener process')
    parser.add_argument('--sigmav', default=DEFAULT_SIGMA_V, type=float,
                        help='scaling factor for the measurement Wiener process')
    parser.add_argument('--dt', default=DEFAULT_DT, type=float, help='sampling time')
    parser.add_argument('--max_d_theta_dt_norm', default=DEFAULT_MAX_D_THETA_DT_NORM, type=float, help='maximum allowed d_theta_dt_norm')
    parser.add_argument('--min_fisher_ev', default=DEFAULT_MIN_FISHER_EV, type=float, help='minimum eigenvalue for Fisher truncation')
    parser.add_argument('--tf', default=DEFAULT_FINAL_TIME, type=float, help='final simulation time')
    parser.add_argument('--varscale', default=DEFAULT_VAR_SCALE, type=float,
                        help='Scalar multiplier to identity for initial variance'
                        )
    parser.add_argument('--scale', default=DEFAULT_PAR_SCALE, type=float, help='bijection parameter scale factor')
    parser.add_argument('--mean_scale', default=DEFAULT_MEAN_SCALE, type=float, help='mean scale factor')
    parser.add_argument('--n_meas', default=DEFAULT_N_MEAS, type=int, help='how many measurements are taken')
    parser.add_argument('--mmt_iter', default=DEFAULT_MMT_ITER, type=int,
                        help='moment_matching_iteration_number')
    parser.add_argument('--n_particle', default=DEFAULT_N_PARTICLE, type=int,
                        help='number of particle for particle filter for each '
                             'processing device')
    parser.add_argument('--seed', default=DEFAULT_SEED, type=int, help='prngkey for jax.random')
    parser.add_argument('--s_level', default=DEFAULT_S_LEVEL, type=int, help='sparse grid integration level')
    parser.add_argument('--n_point', default=DEFAULT_N_POINT, type=int, help='number of particles per dimension')
    parser.add_argument('--max_order_monomials', default=DEFAULT_MAX_ORDER, type=int,
                        help='maximum total order of monomials in the natural statistics')
    parser.add_argument('--pgm_samples', default=DEFAULT_N_PGM_SMPLS, type=int, help='Number of pgm samples')
    parser.add_argument('--pgm_clusters', default=DEFAULT_PGM_CLSTR, type=int, help='Number of pgm clusters')
    parser.add_argument('--sim_case', default=1, type=int, help='1 - Projection Filter Only, '
                                                                '2 - Benchmark Only, 3 - Full')

    parser.add_argument('--resampling', default=DEFAULT_RESAMPLING, type=str,
                        help='resampling method for particle filter')
    parser.add_argument('--srule', default=DEFAULT_SPG_RULE, type=str, help='sparse grid integration rule')
    add_boolean_argument(parser, "multicore", default=DEFAULT_MLTCORE,
                         messages="whether to consider each core as separate processing unit")
    add_boolean_argument(parser, "save", default=DEFAULT_SAVE, messages="whether to save the variables to a hdf file.")
    add_boolean_argument(parser, "progress_bar", default=DEFAULT_PROGRESS_BAR, messages="whether to show the progress bar.")
    add_boolean_argument(parser, "constant_step_size", default=DEFAULT_CONSTANT_STEP_SIZE,
                         messages="whether to use a constant step size in the ode solver.")

    args = parser.parse_args()

    #set simulation path
    simulation_time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    simulation_id = "VDP_SEED_{}_".format(args.seed) + simulation_time_string
    sim_case = Simulation_Case(args.sim_case)
    if sim_case == Simulation_Case.PROJECTION_FILTER_ONLY:
        sim_path = pathlib.Path(f'./simulation_result/VDP-{args.max_order_monomials}-{str(sim_case)}-F_MIN_EV_{args.min_fisher_ev}-MAX_DTHETA_DT_{args.max_d_theta_dt_norm}-TF_{args.tf}-NMEAS_{args.n_meas}/{simulation_id}')
    else:
        sim_path = pathlib.Path(f'./simulation_result/VDP-{args.max_order_monomials}-{str(sim_case)}-TF_{args.tf}-NMEAS_{args.n_meas}/{simulation_id}')
    if not sim_path.exists():
        sim_path.mkdir(parents=True, exist_ok=True)

    #set configs
    sim_config = SimulationConfig(dt=args.dt,t_f=args.tf,mean_scale=args.mean_scale,n_meas=args.n_meas,seed=args.seed,
                                  n_point_per_axis=args.n_point,n_std=6,use_linear_meas=True,use_multi_core_cpu=args.multicore,
                                  save=args.save,gauss_mixture_clusters=args.pgm_clusters,gauss_mixture_samples=args.pgm_samples,
                                  em_max_iterations=10,sim_path=sim_path,sim_case=sim_case)

    par_config = ParticleFilterConfig(n_particle_per_device=args.n_particle,
                                      resampling=args.resampling,
                                      use_stratonovich=True,
                                      sde_solver=dfx.EulerHeun())
    enkf_config = EnKFConfig(n_particle_per_device=args.n_particle, )


    # set some variables
    kappa_vdp = args.kappa
    mu_vdp = args.mu
    sigma_w_vdp = args.sigmaw
    sigma_v_vdp = args.sigmav
    max_order_monomials = args.max_order_monomials

    meas_dim = 2
    R = sigma_v_vdp ** 2 * jnp.eye(meas_dim)
    R_inv = jnp.linalg.inv(R)
    dynamic_parameters = {
        "kappa_vdp": kappa_vdp,
        "mu_vdp": mu_vdp,
        "sigma_w": sigma_w_vdp,
        "sigma_v": sigma_v_vdp,
    }

    if sim_config.use_multi_core_cpu:
        n_devices = jax.local_device_count()
    else:
        n_devices = 1

    # set natural statistics
    x_sp, dw_sp, dv_sp = sp.symbols(('x1:3', 'dw1:3', 'dv1:3'))
    t_sp = sp.symbols('t')

    f = sp.Matrix([kappa_vdp * x_sp[0] + x_sp[1],
                   kappa_vdp * x_sp[1] + mu_vdp * (1. - x_sp[0] * x_sp[0]) * x_sp[1] - x_sp[0]])
    g = sp.Matrix([[0.], [sigma_w_vdp]])
    dynamic_sde = SDE(f, g, t_sp, x_sp, dw_sp)




    natural_statistics_symbolic_original = mix_monomials_up_to_order(x_sp, max_order_monomials)

    ch_symbolic = sp.Matrix([sp.sin(x_sp[0]), sp.sin(x_sp[1])])
    ch2_symbolic = sp.kronecker_product(ch_symbolic, ch_symbolic)
    ch2_symbolic_set = [
                sp.sin(x_sp[0]) * sp.sin(x_sp[1]),
                sp.sin(x_sp[1])**2,
                sp.sin(x_sp[0])**2
            ]
    # add sin and cos
    stats = []
    for a_stat in natural_statistics_symbolic_original.tolist():
        stats.append(a_stat[0])

    for a_stat in ch_symbolic.tolist():
        stats.append(a_stat[0])

    for a_stat in ch2_symbolic_set:
        stats.append(a_stat)
    natural_statistics_symbolic = sp.Matrix(stats)


    @filter_jit
    @partial(jnp.vectorize, signature='(n)->(n)', excluded=(0, 2))
    def drift(_t: float, _x: jnp.ndarray, args) -> jnp.ndarray:
        """
        the drift part of the dynamic.
        Parameters
        ----------
        _x : jnp.ndarray
            state

        Returns
        -------
        f : jnp.ndarray
            result
        """
        return jnp.array([kappa_vdp * _x[0] + _x[1],
                          kappa_vdp * _x[1] + mu_vdp * (1. - _x[0] * _x[0]) * _x[1] - _x[0]])


    @filter_jit
    @partial(jnp.vectorize, signature='(n)->(n,m)', excluded=(0, 2))
    def diffusion(_t: float, _x: jnp.ndarray, args) -> jnp.ndarray:
        """
        the diffusive part of the dynamic.
        Parameters
        ----------
        _x : jnp.ndarray
            state

        Returns
        -------
        g : jnp.ndarray
            result
        """
        return jnp.array([[0.],
                          sigma_w_vdp * jnp.array([1.])])


    @filter_jit
    @partial(jnp.vectorize, signature='(d)->(m)')
    def meas_fun(_x: jnp.ndarray) -> jnp.ndarray:
        """
        the drift part of the measurement.
        Parameters
        ----------
        _x : jnp.ndarray
            state

        Returns
        -------
        h : jnp.ndarray
            result
        """
        # WARNING: this is hard coded
        return jnp.array([jnp.sin(_x[0]),jnp.sin(_x[1])])


    @filter_jit
    @partial(jnp.vectorize, signature='(d)->()', excluded=(1,))
    def ell_y(_x: jnp.ndarray, _meas: jnp.ndarray) -> jnp.ndarray:
        _h_eval = meas_fun(_x)
        return 0.5 * ((_meas - _h_eval).T @ R_inv @ (_meas - _h_eval))



    #prepare dynamical system
    dynamic = DynamicalSystem(dim_states=2,
                              dim_output=meas_dim,
                              dim_process_brownian=1,
                              dim_meas_brownian=2,
                              sde=dynamic_sde,
                              drift=drift,
                              diffusion=diffusion,
                              measurement=meas_fun,
                              negative_log_likelihood=ell_y,
                              parameters=dynamic_parameters
                              )

    # preparing ProjectionFilterConfig
    T1, T2 = get_T_matrices(ch_symbolic, natural_statistics_symbolic)
    # this is hard coded, which might be bad.
    H = jnp.array([[1., 0., ],
                   [0., 1., ]]).T
    theta_ell_args = T1.T @ H @ R_inv, 0.5 * T2.T @ (H @ R_inv @ H.T).flatten()

    #theta init
    # create mixture Gaussian initial samples
    mean_1 = sim_config.mean_scale * jnp.array([1., -1.])
    mean_2 = -mean_1
    means_init = jnp.array([mean_1, mean_2])
    cov_init = (sim_config.model_scale ** 2) * sim_config.var_scale * jnp.eye(dynamic.dim_states)
    covs_init = jnp.array([cov_init, cov_init])
    gmm = gm.GaussianMixture.from_mean_cov_component_weights(means_init, covs_init,
                                                             jnp.array([0.5, 0.5]))

    prng_key = jrandom.PRNGKey(sim_config.seed)
    prng_key, sub_key = jrandom.split(prng_key)
    init_samples = gmm.sample(sub_key, n_devices * par_config.n_particle_per_device)
    init_samples = init_samples.reshape((n_devices, par_config.n_particle_per_device, dynamic.dim_states))
    init_mean = 0.5 * (mean_1 + mean_2)

    var_init_inv = jnp.linalg.solve(cov_init, jnp.eye(dynamic.dim_states))
    gaussian_initial_condition_original, theta_indices_for_bijection_params = (
        calculate_gaussian_natural_parameters(init_mean,
                                              cov_init,
                                              natural_statistics_symbolic_original,
                                              dynamic.sde.variables, ))
    additional_stat_length = len(natural_statistics_symbolic) - len(natural_statistics_symbolic_original)
    gaussian_initial_condition = jnp.concatenate(
        [gaussian_initial_condition_original, jnp.zeros((additional_stat_length,))], )
    _a_scale = 1 / sim_config.var_scale
    _a_mean = init_mean

    # cholesky version is used
    intial_bijection_parameters_gaussian = (init_mean, jnp.linalg.cholesky(cov_init), args.scale)
    rtol = DEFAULT_RTOL  # default 1e-3
    atol = DEFAULT_ATOL
    proj_config = ProjectionFilterConfig(theta_ell_args=theta_ell_args,
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
    means_gm_init = jnp.array([mean_1, mean_2, *[onp.random.randn(2, ) for i in range(sim_config.gauss_mixture_clusters
                                                                                      - 2)]])
    covs_gm_init = jnp.array([cov_init for i in range(means_gm_init.shape[0])])
    small_probability = 1e-2 / sim_config.gauss_mixture_clusters
    weights_gm_init = jnp.array([0.5, 0.5, *[small_probability for
                                             i in range(sim_config.gauss_mixture_clusters - 2)]])
    weights_gm_init = weights_gm_init / jnp.sum(weights_gm_init)

    # prepare logger
    logger = logging.getLogger("VDP")
    log_file = sim_path / "simulation.log"
    handler = logging.FileHandler(log_file)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.info(f"Starting simulation with SEED ={args.seed}")
    progress_bar_type = ""
    if args.progress_bar:
        progress_bar_type = "text"

    if sim_config.sim_case == Simulation_Case.PROJECTION_FILTER_ONLY:
        simulation_result = simulate_projection_filter_only(dynamic,
                                           sim_config,
                                           proj_config,
                                           init_state,
                                           init_samples,
                                           natural_statistics_symbolic,
                                           logger,
                                           progress_bar_type,
                                           )
    elif sim_config.sim_case == Simulation_Case.BENCHMARK_ONLY:
        simulation_result = simulate_benchmark(dynamic,
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
                                               logger
                                               )
    elif sim_config.sim_case == Simulation_Case.FULL:
        simulation_result = simulate_fully(dynamic,
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
                                           progress_bar_type
                                           )
    else:
        print("Unknown simulation case")

