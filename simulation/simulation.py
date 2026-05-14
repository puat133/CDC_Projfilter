import logging
import multiprocessing
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count())

import jax
jax.config.update("jax_enable_x64", True)  # this must be enabled, or the particle filter simulation will be wrong
DEBUG_MODE = False
if DEBUG_MODE:
    jax.config.update('jax_disable_jit', True)
    jax.config.update("jax_debug_nans", False)

import jax.numpy as jnp
import sympy as sp
import jax.random as jrandom
from cd_filtering.cd_proj_conjugate import proj_error_norm_square_hist_scan
from exponential_family.n_d_ef_spg import NDExponentialFamilySPG
from utils.hdf_io import save_to_hdf
import diffrax as dfx
from simulation.configs import SimulationConfig, EnKFConfig, ParticleFilterConfig, ProjectionFilterConfig, HGMFConfig
from simulation.dynamical_system import DynamicalSystem
from simulation.post_analysis import post_analysis
import simulation.runs as runs
from jaxtyping import Array
# Enable debugging
jax.debug_infs()
jax.debug_nans()

def simulate_projection_filter_only(
        dynamic:DynamicalSystem,
        sim_config:SimulationConfig,
        proj_config:ProjectionFilterConfig,
        init_state:Array,
        init_samples:Array,
        natural_statistics_symbolic: sp.Matrix,
        logger:logging.Logger,
        progress_bar_type:str,
        sde_solver: dfx.AbstractSolver=dfx.ShARK(),# only for additive noise
        recompute_init_state: bool=True,
        simplify_lc: bool=True,
)-> dict:
    """
    Simulate the dynamical system using various filtering techniques.

    Parameters
    ----------
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    sim_config : SimulationConfig
        Configuration parameters for the simulation.
    proj_config : ProjectionFilterConfig
        Configuration parameters for the projection filter.
    par_config : ParticleFilterConfig
        Configuration parameters for the particle filter.
    enkf_config : EnKFConfig
        Configuration parameters for the EnKF.
    init_state : Array
        Initial state of the system.
    init_samples : Array
        Initial samples for the particles.
    natural_statistics_symbolic : sp.Matrix
        Symbolic representation of the natural statistics.
    means_gm_init : Array
        Initial means for the Gaussian mixture model.
    covs_gm_init : Array
        Initial covariances for the Gaussian mixture model.
    weights_gm_init : Array
        Initial weights for the Gaussian mixture model.
    logger : logging.Logger
        Logger for logging information.
    progress_bar_type : str, optional
        Whether to show progress bar during computation, defaults to no progress bar. The other options are text, and web.

    Returns
    -------
    dict
        A dictionary containing the results of the simulation.
    """
    try:
        logger.info("Initialization ...")

        if sim_config.use_multi_core_cpu:
            n_devices = jax.local_device_count()
        else:
            n_devices = 1

        # classes that will be saved to hdf file
        classes = [NDExponentialFamilySPG,
                   SimulationConfig,
                   EnKFConfig,
                   ParticleFilterConfig,
                   ProjectionFilterConfig,
                   DynamicalSystem,
                   dfx.Solution,
                   list,
                   tuple,
                   dict]

        t_s = 0
        t_span = jnp.linspace(t_s, sim_config.t_f, sim_config.n_meas + 1)
        prng_key = jrandom.PRNGKey(sim_config.seed)
        prng_key, sub_key = jrandom.split(prng_key)
        x_integrated, measurement_record, t_meas, time_sample = dynamic.generate_path_and_measurement(init_state,
                                                                                                      t_span,
                                                                                                      sim_config.dt,
                                                                                                      sub_key,
                                                                                                      sde_solver)

        logger.info("Preparing projection filter simulation(s)...")
        (theta_init, params_init,
         psi_theta_init, p, natural_statistics, lc_jax) = runs.prepare_projection_filter(dynamic,
                                                                                         proj_config,
                                                                                         natural_statistics_symbolic,
                                                                                         init_samples,
                                                                                         proj_config.theta_init,
                                                                                         proj_config.params_init, 
                                                                                         recompute_init_state,
                                                                                         simplify_lc)


        logger.info("Running Projection Filter...")
        cdf_results, cd_proj_filter_cost_analysis, execution_time_cdf = runs.run_projection_filter(sim_config,
                                                                                                   proj_config,
                                                                                                   p,
                                                                                                   lc_jax,
                                                                                                   theta_init,
                                                                                                   params_init,
                                                                                                   measurement_record,
                                                                                                   progress_bar_type
                                                                                                   )

        logger.info(
            "Completed running projection filter. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_cdf.total_seconds(),
                cd_proj_filter_cost_analysis["corrected_flops"]
            ))
        nan_presence = jnp.any(jnp.isnan(cdf_results[0]))
        logger.info("Nan values presence in cdf_results is {}".format(nan_presence))

        if proj_config.save_fokker_planck_thetas:
            # include intermediate step as well:
            shape_ts = cdf_results[5].ts.shape
            shape_ys_0 = cdf_results[5].ys[0].shape
            shape_ys_1 = cdf_results[5].ys[1].shape
            shape_ys_2 = cdf_results[5].ys[2].shape

            theta_hist = jnp.reshape(cdf_results[5].ys[0],(shape_ys_0[0]*shape_ys_0[1],shape_ys_0[2]))
            params_0_hist = jnp.reshape(cdf_results[5].ys[1],(shape_ys_1[0]*shape_ys_1[1],shape_ys_1[2]))
            params_1_hist = jnp.reshape(cdf_results[5].ys[2], (shape_ys_2[0] * shape_ys_2[1], shape_ys_2[2],
                                                               shape_ys_2[3]
                                                               ))

            params_2_hist = jnp.ones(shape_ys_2[0] * shape_ys_2[1],)*proj_config.par_scale
            params_hist = (params_0_hist, params_1_hist, params_2_hist)
            t_s = jnp.reshape(cdf_results[5].ts,(shape_ts[0]*shape_ts[1],))

            sqr_error_norm_hists = proj_error_norm_square_hist_scan(p,
                                                                    natural_statistics_symbolic,
                                                                    lc_jax,
                                                                    dynamic,
                                                                    theta_hist,
                                                                    params_hist)

        else:
            sqr_error_norm_hists = proj_error_norm_square_hist_scan(p,
                                                                    natural_statistics_symbolic,
                                                                    lc_jax,
                                                                    dynamic,
                                                                    cdf_results[0],
                                                                    cdf_results[1])
            logger.info("Doing post analysis.")
        results = post_analysis(
                                dynamic,
                                sim_config,
                                p,
                                measurement_record,
                                theta_init,
                                params_init,
                                cdf_results,
                                logger,
                                natural_statistics_symbolic
                                )

        if sim_config.save and not nan_presence:
            logger.info("Saving variables to hdf file...")
            save_to_hdf(sim_config.sim_path / "variables.hdf",
                        locals(),
                        classes=classes,
                        excluded_vars=['x_particle_posterior_history',
                                       'x_particle_prior_history',
                                       'enkf_particle_posterior_history',
                                       'enkf_samples_post',
                                       'res_cd_enkf',
                                       'samples_post',
                                       'samples_pred',
                                       'samples',
                                       'samples_1',
                                       'samples_2',
                                       'a_samples',
                                       'a_init_samples',
                                       'bijected_points_pred',
                                       'bijected_points_post',
                                       'cd_par_filt_results',
                                       'init_samples',
                                       'log_weights_history'
                                       ])

        logger.info("Completed. Enjoy.")
    except Exception as e:
        logger.exception(str(e))
        raise


    return {"results":0}


def simulate_benchmark(
        dynamic:DynamicalSystem,
        sim_config:SimulationConfig,
        proj_config:ProjectionFilterConfig,
        par_config:ParticleFilterConfig,
        enkf_config:EnKFConfig,
        init_state:Array,
        init_samples:Array,
        natural_statistics_symbolic: sp.Matrix,
        means_gm_init: Array,
        covs_gm_init: Array,
        weights_gm_init: Array,
        logger:logging.Logger,
        hgmf_config: HGMFConfig | None = None,
)-> dict:
    """
    Benchmark filtering techniques without running the projection filter.

    This function runs all filtering methods except the projection filter,
    making it useful for benchmarking non-projection filtering approaches.

    Parameters
    ----------
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    sim_config : SimulationConfig
        Configuration parameters for the simulation.
    proj_config : ProjectionFilterConfig
        Configuration parameters for the projection filter (used for initialization only).
    par_config : ParticleFilterConfig
        Configuration parameters for the particle filter.
    enkf_config : EnKFConfig
        Configuration parameters for the EnKF.
    init_state : Array
        Initial state of the system.
    init_samples : Array
        Initial samples for the particles.
    natural_statistics_symbolic : sp.Matrix
        Symbolic representation of the natural statistics.
    means_gm_init : Array
        Initial means for the Gaussian mixture model.
    covs_gm_init : Array
        Initial covariances for the Gaussian mixture model.
    weights_gm_init : Array
        Initial weights for the Gaussian mixture model.
    logger : logging.Logger
        Logger for logging information.

    Returns
    -------
    dict
        A dictionary containing the results of the simulation.
    """
    try:
        logger.info("Initialization ...")

        if sim_config.use_multi_core_cpu:
            n_devices = jax.local_device_count()
        else:
            n_devices = 1

        # classes that will be saved to hdf file
        classes = [NDExponentialFamilySPG,
                   SimulationConfig,
                   EnKFConfig,
                   ParticleFilterConfig,
                   ProjectionFilterConfig,
                   DynamicalSystem,
                   dfx.Solution,
                   list,
                   tuple,
                   dict]

        t_s = 0
        t_span = jnp.linspace(t_s, sim_config.t_f, sim_config.n_meas + 1)
        prng_key = jrandom.PRNGKey(sim_config.seed)
        prng_key, sub_key = jrandom.split(prng_key)
        x_integrated, measurement_record, t_meas, time_sample = dynamic.generate_path_and_measurement(init_state,
                                                                                                      t_span,
                                                                                                      sim_config.dt,
                                                                                                      sub_key)

        # Prepare exponential density (needed for post_analysis)
        _, _, _, p, _, _ = runs.prepare_projection_filter(dynamic, proj_config, natural_statistics_symbolic,
                                                           init_samples, proj_config.theta_init,
                                                           proj_config.params_init)

        logger.info("Running particle filter simulation...")
        prng_key, sub_key = jrandom.split(prng_key)
        (x_particle_posterior_history, x_particle_prior_history, cd_particle_filter_cost_analysis,
         execution_time_cd_par_filt) = runs.run_particle_filter(sim_config,
                                                                dynamic,
                                                                measurement_record,
                                                                par_config,
                                                                init_samples,
                                                                sub_key)

        logger.info(
            "Completed running particle filter. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_cd_par_filt.total_seconds(),
                cd_particle_filter_cost_analysis["corrected_flops"]
            ))

        logger.info("Running EnKF simulation...")
        prng_key, sub_key = jrandom.split(prng_key)
        enkf_particle_posterior_history, enkf_cost_analysis, execution_time_enkf = runs.run_enkf(sim_config,
                                                                                                 dynamic,
                                                                                                 measurement_record,
                                                                                                 enkf_config,
                                                                                                 init_samples,
                                                                                                 sub_key, )
        logger.info(
            "Completed running EnKF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_enkf.total_seconds(),
                enkf_cost_analysis["corrected_flops"]
            ))

        logger.info("Preparing GSF,SP-GSF, PGM (K-mean), PGM (EM) simulations...")
        log_weights_gm_init = jnp.log(weights_gm_init)
        cluster_gsf_count = sim_config.gauss_mixture_clusters


        logger.info("Running GSF...")
        gsf_results, gsf_cost_analysis, execution_time_gsf = runs.run_gsf(sim_config,
                                                                          dynamic,
                                                                          measurement_record,
                                                                          means_gm_init,
                                                                          covs_gm_init,
                                                                          log_weights_gm_init,
                                                                          )
        logger.info(
            "Completed running GSF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_gsf.total_seconds(),
                gsf_cost_analysis["corrected_flops"]
            ))

        logger.info("Running SP-GSF...")

        sp_gsf_results, sp_gsf_cost_analysis, execution_time_sp_gsf = runs.run_sp_gsf(sim_config,
                                                                                      dynamic,
                                                                                      measurement_record,
                                                                                      means_gm_init,
                                                                                      covs_gm_init,
                                                                                      log_weights_gm_init,
                                                                                      5
                                                                                      )
        logger.info(
            "Completed running SP-GSF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_sp_gsf.total_seconds(),
                sp_gsf_cost_analysis["corrected_flops"]
            ))

        logger.info("Running PGM (K-Mean)...")
        prng_key, sub_key = jrandom.split(prng_key)
        pgm_results, pgm_cost_analysis, execution_time_pgm = runs.run_pgm(sim_config,
                                                       dynamic,
                                                       measurement_record,
                                                       means_gm_init,
                                                       covs_gm_init,
                                                       weights_gm_init,
                                                       sim_config.gauss_mixture_samples,
                                                       sub_key)
        logger.info(
            "Completed running PGM (K-mean). Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_pgm.total_seconds(),
                pgm_cost_analysis["corrected_flops"]
            ))

        logger.info("Running PGM (EM)...")
        prng_key, sub_key = jrandom.split(prng_key)
        pgm_em_results, pgm_em_cost_analysis, execution_time_pgm_em = runs.run_pgm_em(sim_config,
                                                                dynamic,
                                                                measurement_record,
                                                                means_gm_init,
                                                                covs_gm_init,
                                                                weights_gm_init,
                                                                sim_config.gauss_mixture_samples,
                                                                sub_key,
                                                                sim_config.em_max_iterations)
        logger.info(
            "Completed running PGM (EM). Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_pgm_em.total_seconds(),
                pgm_em_cost_analysis["corrected_flops"]
            ))

        if hgmf_config is None:
            hgmf_config = HGMFConfig()
        logger.info("Running HGMF...")
        hgmf_results, hgmf_cost_analysis, execution_time_hgmf = runs.run_hgmf(
            sim_config,
            dynamic,
            measurement_record,
            means_gm_init,
            covs_gm_init,
            log_weights_gm_init,
            hgmf_config,
        )
        logger.info(
            "Completed running HGMF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_hgmf.total_seconds(),
                hgmf_cost_analysis["corrected_flops"]
            ))

        logger.info("Doing post analysis.")
        results = post_analysis(dynamic,
                                sim_config,
                                p,
                                measurement_record,
                                proj_config.theta_init,
                                proj_config.params_init,
                                None,  # cdf_results - not computed in benchmark mode
                                logger,
                                natural_statistics_symbolic,
                                par_config=par_config,
                                init_samples=init_samples,
                                n_devices=n_devices,
                                means_gm_init=means_gm_init,
                                covs_gm_init=covs_gm_init,
                                log_weights_gm_init=log_weights_gm_init,
                                x_particle_posterior_history=x_particle_posterior_history,
                                x_particle_prior_history=x_particle_prior_history,
                                enkf_particle_posterior_history=enkf_particle_posterior_history,
                                gsf_results=gsf_results,
                                sp_gsf_results=sp_gsf_results,
                                pgm_results=pgm_results,
                                pgm_em_results=pgm_em_results,
                                hgmf_results=hgmf_results,
                                x_meas=x_integrated[1:]
                                )


        if sim_config.save:
            logger.info("Saving variables to hdf file...")
            save_to_hdf(sim_config.sim_path/"variables.hdf",
                        locals(),
                        classes=classes,
                        excluded_vars=[
                                    #    'x_particle_posterior_history',
                                       'x_particle_prior_history',
                                       'enkf_particle_posterior_history',
                                       'enkf_samples_post',
                                       'res_cd_enkf',
                                       'samples_post',
                                       'samples_pred',
                                       'samples',
                                       'samples_1',
                                       'samples_2',
                                       'a_samples',
                                       'a_init_samples',
                                       'bijected_points_pred',
                                       'bijected_points_post',
                                       'cd_par_filt_results',
                                    #    'init_samples',
                                       'log_weights_history'
                                       ])

        logger.info("Completed. Enjoy.")
    except Exception as e:
        logger.exception(str(e))
        raise

    return results


def simulate_fully(
        dynamic:DynamicalSystem,
        sim_config:SimulationConfig,
        proj_config:ProjectionFilterConfig,
        par_config:ParticleFilterConfig,
        enkf_config:EnKFConfig,
        init_state:Array,
        init_samples:Array,
        natural_statistics_symbolic: sp.Matrix,
        means_gm_init: Array,
        covs_gm_init: Array,
        weights_gm_init: Array,
        logger:logging.Logger,
        progress_bar_type:str,
        hgmf_config: HGMFConfig | None = None,
)-> dict:
    """
    Simulate the dynamical system using various filtering techniques.

    Parameters
    ----------
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    sim_config : SimulationConfig
        Configuration parameters for the simulation.
    proj_config : ProjectionFilterConfig
        Configuration parameters for the projection filter.
    par_config : ParticleFilterConfig
        Configuration parameters for the particle filter.
    enkf_config : EnKFConfig
        Configuration parameters for the EnKF.
    init_state : Array
        Initial state of the system.
    init_samples : Array
        Initial samples for the particles.
    natural_statistics_symbolic : sp.Matrix
        Symbolic representation of the natural statistics.
    means_gm_init : Array
        Initial means for the Gaussian mixture model.
    covs_gm_init : Array
        Initial covariances for the Gaussian mixture model.
    weights_gm_init : Array
        Initial weights for the Gaussian mixture model.
    logger : logging.Logger
        Logger for logging information.
    progress_bar_type : str, optional
        Whether to show progress bar during computation, defaults to no progress bar. The other options are text, and web.

    Returns
    -------
    dict
        A dictionary containing the results of the simulation.
    """
    try:
        logger.info("Initialization ...")

        if sim_config.use_multi_core_cpu:
            n_devices = jax.local_device_count()
        else:
            n_devices = 1

        # classes that will be saved to hdf file
        classes = [NDExponentialFamilySPG,
                   SimulationConfig,
                   EnKFConfig,
                   ParticleFilterConfig,
                   ProjectionFilterConfig,
                   DynamicalSystem,
                   dfx.Solution,
                   list,
                   tuple,
                   dict]




        t_s = 0
        t_span = jnp.linspace(t_s, sim_config.t_f, sim_config.n_meas + 1)
        prng_key = jrandom.PRNGKey(sim_config.seed)
        prng_key, sub_key = jrandom.split(prng_key)
        x_integrated, measurement_record, t_meas, time_sample = dynamic.generate_path_and_measurement(init_state,
                                                                                                      t_span,
                                                                                                      sim_config.dt,
                                                                                                      sub_key)

        logger.info("Preparing projection filter simulation(s)...")
        (theta_init, params_init,
         psi_theta_init, p, natural_statistics, lc_jax) = runs.prepare_projection_filter(dynamic,
                                                                                         proj_config,
                                                                                         natural_statistics_symbolic,
                                                                                         init_samples,
                                                                                         proj_config.theta_init,
                                                                                         proj_config.params_init,)


        logger.info("Running Projection Filter...")
        cdf_results, cd_proj_filter_cost_analysis, execution_time_cdf = runs.run_projection_filter(sim_config,
                                                                                                   proj_config,
                                                                                                   p,
                                                                                                   lc_jax,
                                                                                                   theta_init,
                                                                                                   params_init,
                                                                                                   measurement_record,
                                                                                                   progress_bar_type
                                                                                                   )

        logger.info(
            "Completed running projection filter. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_cdf.total_seconds(),
                cd_proj_filter_cost_analysis["corrected_flops"]
            ))
        logger.info("Nan values presence in cdf_results is {}".format(jnp.any(jnp.isnan(cdf_results[0]))))
        if proj_config.save_fokker_planck_thetas:
            # include intermediate step as well:
            shape_ts = cdf_results[5].ts.shape
            shape_ys_0 = cdf_results[5].ys[0].shape
            shape_ys_1 = cdf_results[5].ys[1].shape
            shape_ys_2 = cdf_results[5].ys[2].shape

            theta_hist = jnp.reshape(cdf_results[5].ys[0], (shape_ys_0[0] * shape_ys_0[1], shape_ys_0[2]))
            params_0_hist = jnp.reshape(cdf_results[5].ys[1], (shape_ys_1[0] * shape_ys_1[1], shape_ys_1[2]))
            params_1_hist = jnp.reshape(cdf_results[5].ys[2], (shape_ys_2[0] * shape_ys_2[1], shape_ys_2[2],
                                                               shape_ys_2[3]
                                                               ))

            params_2_hist = jnp.ones(shape_ys_2[0] * shape_ys_2[1], ) * proj_config.par_scale
            params_hist = (params_0_hist, params_1_hist, params_2_hist)
            t_s = jnp.reshape(cdf_results[5].ts, (shape_ts[0] * shape_ts[1],))

            sqr_error_norm_hists = proj_error_norm_square_hist_scan(p,
                                                                    natural_statistics_symbolic,
                                                                    lc_jax,
                                                                    dynamic,
                                                                    theta_hist,
                                                                    params_hist)

        else:
            sqr_error_norm_hists = proj_error_norm_square_hist_scan(p,
                                                                    natural_statistics_symbolic,
                                                                    lc_jax,
                                                                    dynamic,
                                                                    cdf_results[0],
                                                                    cdf_results[1])


        logger.info("Running particle filter simulation...")
        prng_key, sub_key = jrandom.split(prng_key)
        (x_particle_posterior_history, x_particle_prior_history, cd_particle_filter_cost_analysis,
         execution_time_cd_par_filt) = runs.run_particle_filter(sim_config,
                                                                dynamic,
                                                                measurement_record,
                                                                par_config,
                                                                init_samples,
                                                                sub_key)

        logger.info(
            "Completed running particle filter. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_cd_par_filt.total_seconds(),
                cd_particle_filter_cost_analysis["corrected_flops"]
            ))

        logger.info("Running EnKF simulation...")
        prng_key, sub_key = jrandom.split(prng_key)
        enkf_particle_posterior_history, enkf_cost_analysis, execution_time_enkf = runs.run_enkf(sim_config,
                                                                                                 dynamic,
                                                                                                 measurement_record,
                                                                                                 enkf_config,
                                                                                                 init_samples,
                                                                                                 sub_key, )
        logger.info(
            "Completed running EnKF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_enkf.total_seconds(),
                enkf_cost_analysis["corrected_flops"]
            ))

        logger.info("Preparing GSF,SP-GSF, PGM (K-mean), PGM (EM) simulations...")
        log_weights_gm_init = jnp.log(weights_gm_init)
        cluster_gsf_count = sim_config.gauss_mixture_clusters


        logger.info("Running GSF...")
        gsf_results, gsf_cost_analysis, execution_time_gsf = runs.run_gsf(sim_config,
                                                                          dynamic,
                                                                          measurement_record,
                                                                          means_gm_init,
                                                                          covs_gm_init,
                                                                          log_weights_gm_init,
                                                                          )
        logger.info(
            "Completed running GSF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_gsf.total_seconds(),
                gsf_cost_analysis["corrected_flops"]
            ))

        logger.info("Running SP-GSF...")

        sp_gsf_results, sp_gsf_cost_analysis, execution_time_sp_gsf = runs.run_sp_gsf(sim_config,
                                                                                      dynamic,
                                                                                      measurement_record,
                                                                                      means_gm_init,
                                                                                      covs_gm_init,
                                                                                      log_weights_gm_init,
                                                                                      5
                                                                                      )
        logger.info(
            "Completed running SP-GSF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_sp_gsf.total_seconds(),
                sp_gsf_cost_analysis["corrected_flops"]
            ))

        logger.info("Running PGM (K-Mean)...")
        prng_key, sub_key = jrandom.split(prng_key)
        pgm_results, pgm_cost_analysis, execution_time_pgm = runs.run_pgm(sim_config,
                                                       dynamic,
                                                       measurement_record,
                                                       means_gm_init,
                                                       covs_gm_init,
                                                       weights_gm_init,
                                                       sim_config.gauss_mixture_samples,
                                                       sub_key)
        logger.info(
            "Completed running PGM (K-mean). Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_pgm.total_seconds(),
                pgm_cost_analysis["corrected_flops"]
            ))

        logger.info("Running PGM (EM)...")
        prng_key, sub_key = jrandom.split(prng_key)
        pgm_em_results, pgm_em_cost_analysis, execution_time_pgm_em = runs.run_pgm_em(sim_config,
                                                                dynamic,
                                                                measurement_record,
                                                                means_gm_init,
                                                                covs_gm_init,
                                                                weights_gm_init,
                                                                sim_config.gauss_mixture_samples,
                                                                sub_key,
                                                                sim_config.em_max_iterations)
        logger.info(
            "Completed running PGM (EM). Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_pgm_em.total_seconds(),
                pgm_em_cost_analysis["corrected_flops"]
            ))

        if hgmf_config is None:
            hgmf_config = HGMFConfig()
        logger.info("Running HGMF...")
        hgmf_results, hgmf_cost_analysis, execution_time_hgmf = runs.run_hgmf(
            sim_config,
            dynamic,
            measurement_record,
            means_gm_init,
            covs_gm_init,
            log_weights_gm_init,
            hgmf_config,
        )
        logger.info(
            "Completed running HGMF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_hgmf.total_seconds(),
                hgmf_cost_analysis["corrected_flops"]
            ))

        logger.info("Doing post analysis.")
        results = post_analysis(dynamic,
                                sim_config,
                                p,
                                measurement_record,
                                theta_init,
                                params_init,
                                cdf_results,
                                logger,
                                natural_statistics_symbolic,
                                par_config=par_config,
                                init_samples=init_samples,
                                n_devices=n_devices,
                                means_gm_init=means_gm_init,
                                covs_gm_init=covs_gm_init,
                                log_weights_gm_init=log_weights_gm_init,
                                x_particle_posterior_history=x_particle_posterior_history,
                                x_particle_prior_history=x_particle_prior_history,
                                enkf_particle_posterior_history=enkf_particle_posterior_history,
                                gsf_results=gsf_results,
                                sp_gsf_results=sp_gsf_results,
                                pgm_results=pgm_results,
                                pgm_em_results=pgm_em_results,
                                hgmf_results=hgmf_results,
                                x_meas=x_integrated[1:],
                                theta_indices_for_mean=proj_config.theta_indices_for_bijection_params[0]
                                )


        if sim_config.save:
            logger.info("Saving variables to hdf file...")
            save_to_hdf(sim_config.sim_path/"variables.hdf",
                        locals(),
                        classes=classes,
                        excluded_vars=['x_particle_posterior_history',
                                       'x_particle_prior_history',
                                       'enkf_particle_posterior_history',
                                       'enkf_samples_post',
                                       'res_cd_enkf',
                                       'samples_post',
                                       'samples_pred',
                                       'samples',
                                       'samples_1',
                                       'samples_2',
                                       'a_samples',
                                       'a_init_samples',
                                       'bijected_points_pred',
                                       'bijected_points_post',
                                       'cd_par_filt_results',
                                       'init_samples',
                                       'log_weights_history'
                                       ])

        logger.info("Completed. Enjoy.")
    except Exception as e:
        logger.exception(str(e))
        raise

    return results


def simulate_hgmf_only(
        dynamic: DynamicalSystem,
        sim_config: SimulationConfig,
        init_state: Array,
        init_samples: Array,
        means_gm_init: Array,
        covs_gm_init: Array,
        weights_gm_init: Array,
        logger: logging.Logger,
        hgmf_config: HGMFConfig | None = None,
) -> dict:
    """Run ONLY the Homotopic Gaussian Mixture Filter for the given seed.

    Measurement record is generated deterministically from sim_config.seed
    (same path as simulate_benchmark / simulate_fully), so the resulting
    HGMF posterior trajectory aligns with the existing BENCHMARK_ONLY outputs
    for the same seed and can be compared post-hoc.

    Saves to sim_config.sim_path / "variables.hdf" a payload that contains:
        - the measurement record and ground-truth trajectory,
        - the initial-prior GM parameters (means, covs, log_weights),
        - the HGMF result tuple (means_hist, covs_hist, log_weights_hist),
        - cost analysis, execution time, and the configs.
    """
    classes = [SimulationConfig, HGMFConfig, DynamicalSystem, dfx.Solution, list, tuple, dict]

    try:
        logger.info("Initialization (HGMF-only)...")

        t_s = 0
        t_span = jnp.linspace(t_s, sim_config.t_f, sim_config.n_meas + 1)
        prng_key = jrandom.PRNGKey(sim_config.seed)
        prng_key, sub_key = jrandom.split(prng_key)

        # Same measurement-generation path as simulate_benchmark — same seed
        # therefore yields the same measurement record.
        x_integrated, measurement_record, t_meas, time_sample = \
            dynamic.generate_path_and_measurement(init_state,
                                                  t_span,
                                                  sim_config.dt,
                                                  sub_key)

        if hgmf_config is None:
            hgmf_config = HGMFConfig()

        log_weights_gm_init = jnp.log(weights_gm_init)

        logger.info("Running HGMF...")
        hgmf_results, hgmf_cost_analysis, execution_time_hgmf = runs.run_hgmf(
            sim_config,
            dynamic,
            measurement_record,
            means_gm_init,
            covs_gm_init,
            log_weights_gm_init,
            hgmf_config,
        )
        logger.info(
            "Completed running HGMF. Execution time = {:.3e} seconds - Flops = {:.3e}".format(
                execution_time_hgmf.total_seconds(),
                hgmf_cost_analysis["corrected_flops"]
            ))

        nan_presence = jnp.any(jnp.isnan(hgmf_results[0]))
        logger.info("NaN values presence in HGMF results: {}".format(bool(nan_presence)))

        if sim_config.save and not nan_presence:
            logger.info("Saving variables to hdf file...")
            save_to_hdf(sim_config.sim_path / "variables.hdf",
                        locals(),
                        classes=classes,
                        excluded_vars=['init_samples'])

        logger.info("Completed (HGMF-only). Enjoy.")
        return {
            "hgmf_results": hgmf_results,
            "hgmf_cost_analysis": hgmf_cost_analysis,
            "execution_time_hgmf": execution_time_hgmf,
            "measurement_record": measurement_record,
        }
    except Exception as e:
        logger.exception(str(e))
        raise
