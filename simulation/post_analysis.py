import jax.numpy as jnp
import jax.random as jrandom
import jax
import cd_filtering.bayesian_update as bu
import utils.density_manipulations as dm
import utils.metrics as metrics
from exponential_family.n_d_ef import NDExponentialFamily
from simulation.configs import SimulationConfig, ParticleFilterConfig, Simulation_Case
from simulation.dynamical_system import DynamicalSystem
from jaxtyping import Array, Float
from equinox import filter_jit
import warnings
import logging
import sympy as sp
from typing import Dict, Tuple, Callable, List, Optional, Any, Union

def post_analysis(dynamic: DynamicalSystem,
                 sim_config: SimulationConfig,
                 p: NDExponentialFamily,
                 measurement_record: Array,
                 theta_init: Array,
                 params_init: tuple[Array, Array, Float],
                 cdf_results: Any,
                 logger: logging.Logger,
                 natural_statistics_symbolic: sp.Matrix,
                 par_config: Optional[ParticleFilterConfig] = None,
                 init_samples: Optional[Array] = None,
                 n_devices: Optional[int] = None,
                 means_gm_init: Optional[Array] = None,
                 covs_gm_init: Optional[Array] = None,
                 log_weights_gm_init: Optional[Array] = None,
                 x_particle_posterior_history: Optional[Array] = None,
                 x_particle_prior_history: Optional[Array] = None,
                 enkf_particle_posterior_history: Optional[Array] = None,
                 gsf_results: Optional[Tuple] = None,
                 sp_gsf_results: Optional[Tuple] = None,
                 pgm_results: Optional[Tuple] = None,
                 pgm_em_results: Optional[Tuple] = None,
                 x_meas: Optional[Array] = None,
                 theta_indices_for_mean: Optional[Array] = None,
                 ) -> Dict[str, Array]:
    """Perform post-analysis on filtering results.
    
    This function analyzes filtering results and computes metrics like Hellinger distances 
    and cross-entropy. It selects between projection-filter-only analysis or full comparison
    analysis based on the simulation configuration.
    
    Parameters
    ----------
    dynamic : DynamicalSystem
        The dynamical system model.
    sim_config : SimulationConfig
        Simulation configuration with sim_case determining analysis type.
    p : NDExponentialFamily
        The exponential family distribution.
    measurement_record : Array
        Record of measurements.
    theta_init : Array 
        Initial natural parameters.
    params_init : tuple[Array, Array, Float]
        Initial bijection parameters (location, scale, normalization constant).
    cdf_results : Any
        Results from the projection filter.
    logger : logging.Logger
        Logger for reporting diagnostics.
    natural_statistics_symbolic : sp.Matrix
        Symbolic representation of natural statistics.
    par_config, init_samples, n_devices, etc. : Optional parameters
        Additional parameters required for full analysis.
    
    Returns
    -------
    Dict[str, Array]
        Dictionary of analysis results.
    """
    # Route to appropriate analysis function based on dimension and sim_case
    # Check projection filter only case first (no particle data available)
    if sim_config.sim_case == Simulation_Case.PROJECTION_FILTER_ONLY:
        if dynamic.dim_states <= 2:
            return _post_analysis_for_projection_filter_only(
                sim_config, p, measurement_record, theta_init, params_init,
                cdf_results, logger)
        else:
            return {}
    else:
        if dynamic.dim_states > 2:
            # High-dimensional systems use sample-based Sliced Wasserstein distance
            return _post_analysis_full_for_high_dim(
                dynamic, sim_config, par_config, p, init_samples, measurement_record,
                n_devices, means_gm_init, covs_gm_init, log_weights_gm_init,
                theta_init, params_init, x_particle_posterior_history,
                x_particle_prior_history, enkf_particle_posterior_history,
                cdf_results, gsf_results, sp_gsf_results, pgm_results,
                pgm_em_results, x_meas, theta_indices_for_mean,
                logger,sim_config.save_particle_samples,sim_config.n_sw_projection, sim_config.n_ef_samples)
        else:
            return _post_analysis_full(
                dynamic, sim_config, par_config, p, init_samples, measurement_record,
                n_devices, means_gm_init, covs_gm_init, log_weights_gm_init,
                theta_init, params_init, x_particle_posterior_history,
                x_particle_prior_history, enkf_particle_posterior_history,
                cdf_results, gsf_results, sp_gsf_results, pgm_results,
                pgm_em_results, x_meas, logger)

def _extract_parameters(cdf_results, time_index, theta_init, params_init):
    """Extract theta and params from cdf_results at given time index.
    
    Helper function to extract parameters from projection filter results.
    
    Parameters
    ----------
    cdf_results : Any
        Results from projection filter.
    time_index : int
        Current time index.
    theta_init, params_init : Initial parameters to use when time_index=0
    
    Returns
    -------
    tuple
        theta_post, params_post, theta_pred, params_pred
    """
    if time_index == 0:
        return theta_init, params_init, theta_init, params_init
    
    # Extract posterior parameters
    theta_post = cdf_results[0][time_index - 1]
    params_post = (
        cdf_results[1][0][time_index - 1], 
        cdf_results[1][1][time_index - 1], 
        cdf_results[1][2][time_index - 1]
    )
    
    # Extract prediction parameters based on cdf_results structure
    if len(cdf_results[3]) > 1:
        # Multiple prediction steps saved
        theta_pred = cdf_results[3][0][time_index - 1, -1]
        params_pred = (
            cdf_results[3][1][time_index - 1, -1],
            cdf_results[3][2][time_index - 1, -1],
            params_post[-1]  # Use scale from posterior
        )
    else:
        # Single prediction step per measurement
        theta_pred = cdf_results[3][time_index - 1]
        params_pred = (
            cdf_results[4][0][time_index - 1],
            cdf_results[4][1][time_index - 1],
            cdf_results[4][2][time_index - 1]
        )
    
    return theta_post, params_post, theta_pred, params_pred

def _calculate_grid_limits(params, n_std):
    """Calculate grid limits based on parameters and std multiplier.
    
    Parameters
    ----------
    params : tuple
        Location and scale parameters (mu, chol_cov, _)
    n_std : float
        Number of standard deviations for grid limits
    
    Returns
    -------
    Array
        Grid limits as [[xmin, xmax], [ymin, ymax]]
    """
    mu, chol_cov, _ = params
    max_eigval = jnp.max(jnp.linalg.eigvalsh(chol_cov))
    half_width = 0.5 * max_eigval
    
    centralized_range = n_std * jnp.array([
        [-half_width, half_width],
        [-half_width, half_width]
    ])
    
    return mu[:, jnp.newaxis] + centralized_range

def _post_analysis_for_projection_filter_only(
                                            sim_config: SimulationConfig,
                                            p: NDExponentialFamily,
                                            measurement_record: Array,
                                            theta_init: Array,
                                            params_init: tuple[Array, Array, Float],
                                            cdf_results: Any,
                                            logger: logging.Logger,
                                            ) -> Dict[str, Array]:
    """Perform post-analysis specifically for projection filter only.
    
    Analyzes projection filter results and computes relevant metrics.
    
    Parameters
    ----------
    sim_config : SimulationConfig
        Simulation configuration.
    p : NDExponentialFamily
        Exponential family distribution.
    measurement_record : Array
        Measurement data.
    theta_init, params_init : Initial parameters.
    cdf_results : Results from projection filter.
    logger : Logger for messages.
    dynamic : DynamicalSystem
        Dynamical system model.
    natural_statistic_symbolic : Symbolic natural statistics.
        
    Returns
    -------
    Dict[str, Array]
        Projection filter metrics.
    """
    # Initialize storage for results
    grid_limits_hist_post = []
    grid_limits_hist_pred = []
    grid_hist_post = []
    eta_history_pred = []
    eta_history_post = []
    density_hist_pred = []
    density_hist_post = []

    # Grid resolution
    num_points = jnp.array([sim_config.n_point_per_axis, sim_config.n_point_per_axis], dtype=jnp.int32)
    n_timesteps = measurement_record.shape[0] + 1

    # Process each time step
    for time_index in range(n_timesteps):
        # Extract parameters for this time step
        theta_post, params_post, theta_pred, params_pred = _extract_parameters(
            cdf_results, time_index, theta_init, params_init)
        
        # Calculate grid limits
        grid_limits_post = _calculate_grid_limits(params_post, sim_config.n_std)
        grid_limits_pred = _calculate_grid_limits(params_pred, sim_config.n_std)
        
        # Calculate natural statistics expectations
        eta_history_pred.append(p.natural_statistics_expectation(theta_pred, params_pred))
        eta_history_post.append(p.natural_statistics_expectation(theta_post, params_post))

        # Calculate densities
        grid_pred, density_pred = p.get_density_values(grid_limits_pred, theta_pred, num_points, params_pred)
        grid_post, density_post = p.get_density_values(grid_limits_post, theta_post, num_points, params_post)

        # Store results
        density_hist_post.append(density_post)
        density_hist_pred.append(density_pred)
        grid_limits_hist_post.append(grid_limits_post)
        grid_limits_hist_pred.append(grid_limits_pred)
        grid_hist_post.append(grid_post)
        
    # Stack results
    results = {
        "eta_history_pred": jnp.stack(eta_history_pred),
        "eta_history_post": jnp.stack(eta_history_post),
        "density_hist_pred": jnp.stack(density_hist_pred),
        "density_hist_post": jnp.stack(density_hist_post),
        "grid_limits_hist_post": jnp.stack(grid_limits_hist_post),
        "grid_limits_hist_pred": jnp.stack(grid_limits_hist_pred),
        "grid_hist_post": jnp.stack(grid_hist_post),
    }

    # Check for NaN values in density
    if jnp.any(jnp.isnan(results["density_hist_post"])):
        logger.warning("There is a nan in density_hist_post!")

    return results

def _get_particles_grid_info(samples, sim_config):
    """Calculate grid information from particle samples.
    
    Parameters
    ----------
    samples : Array
        Particle samples.
    sim_config : SimulationConfig
        Simulation configuration.
        
    Returns
    -------
    tuple
        Grid limits, bin edges, and bin spacings.
    """
    # Calculate mean and covariance of particles
    mean_particles = jnp.mean(samples, axis=0)
    cov_particles = jnp.cov(samples, rowvar=False)
    
    # Calculate grid limits
    max_std = jnp.max(jnp.sqrt(jnp.linalg.eigvalsh(cov_particles)))
    half_width = 0.5 * max_std
    
    centralized_range = sim_config.n_std * jnp.array([
        [-half_width, half_width],
        [-half_width, half_width]
    ])
    
    grid_limits = mean_particles[:, jnp.newaxis] + centralized_range
    
    # Calculate grid spacing and bins
    num_points = jnp.array([sim_config.n_point_per_axis, sim_config.n_point_per_axis])
    dxs = jnp.diff(grid_limits, axis=-1).squeeze() / num_points
    
    xbins = jnp.linspace(
        grid_limits[0, 0] - dxs[0], 
        grid_limits[0, 1] + dxs[0],
        sim_config.n_point_per_axis + 1
    )
    
    ybins = jnp.linspace(
        grid_limits[1, 0] - dxs[1], 
        grid_limits[1, 1] + dxs[1],
        sim_config.n_point_per_axis + 1
    )
    
    return grid_limits, (xbins, ybins), dxs

def _calculate_particle_density(samples, bins):
    """Calculate particle density on grid.
    
    Parameters
    ----------
    samples : Array
        Particle samples.
    bins : tuple
        X and Y bin edges.
        
    Returns
    -------
    Array
        Density histogram.
    """
    xbins, ybins = bins
    density, _, _ = jnp.histogram2d(
        samples[:, 0], samples[:, 1],
        bins=[xbins, ybins], 
        density=True
    )
    
    return density.T  # Transpose for consistent orientation

def _post_analysis_full(dynamic: DynamicalSystem,
                       sim_config: SimulationConfig,
                       par_config: ParticleFilterConfig,
                       p: NDExponentialFamily,
                       init_samples: Array,
                       measurement_record: Array,
                       n_devices: int,
                       means_gm_init: Array,
                       covs_gm_init: Array,
                       log_weights_gm_init: Array,
                       theta_init: Array,
                       params_init: tuple[Array, Array, Float],
                       x_particle_posterior_history: Array,
                       x_particle_prior_history: Array,
                       enkf_particle_posterior_history: Array,
                       cdf_results: Any,
                       gsf_results: Any,
                       sp_gsf_results: Any,
                       pgm_results: Any,
                       pgm_em_results: Any,
                       x_meas: Array,
                       logger: logging.Logger) -> Dict[str, Array]:
    """Perform comprehensive post-analysis across all filter types.
    
    Computes comparison metrics between different filtering algorithms.
    
    Parameters
    ----------
    dynamic : DynamicalSystem
        Dynamical system model.
    sim_config : SimulationConfig
        Simulation configuration.
    Various filter parameters and results.
        
    Returns
    -------
    Dict[str, Array]
        Comprehensive comparison metrics.
    """    
    # check if the cdf_results is None
    analyze_cdf_result = True
    if cdf_results is None:
        analyze_cdf_result = False
        logger.warning("CDF results are None, skipping projection filter analysis.")
    
    # Reshape initial samples
    a_init_samples = init_samples.reshape((n_devices * par_config.n_particle_per_device, dynamic.dim_states))

    # Initialize storage for results
    D_half_hist = []
    H_pq_hist = []
    grid_limits_hist_post = []
    grid_limits_hist_pred = []
    grid_hist_post = []
    
    eta_history_pred = []
    eta_history_post = []
    eta_history_pred_particle = []
    eta_history_post_particle = []
    eta_history_enkf = []
    
    cross_entropy_hist_pred = []
    cross_entropy_hist_post = []
    cross_entropy_hist_gsf = []
    cross_entropy_hist_sp_gsf = []
    cross_entropy_hist_pgm = []
    cross_entropy_hist_pgm_2 = []
    
    density_hist_pred = []
    density_hist_post = []
    density_particle_hist_pred = []
    density_particle_hist_post = []
    density_sp_gsf_hist_post = []
    density_gsf_hist_post = []
    density_pgm_hist_post = []
    density_pgm_2_hist_post = []
    density_enkf_hist_post = []
    
    hell_dist_hist_post = []
    hell_dist_hist_pred = []
    hell_dist_hist_gsf = []
    hell_dist_hist_sp_gsf = []
    hell_dist_hist_pgm = []
    hell_dist_hist_pgm_2 = []
    hell_dist_hist_enkf = []

    num_points = jnp.array([sim_config.n_point_per_axis, sim_config.n_point_per_axis], dtype=jnp.int32)
    n_timesteps = measurement_record.shape[0] + 1

    for time_index in range(n_timesteps):
        # Get particle samples for this time step
        if time_index == 0:
            samples_post = a_init_samples
            samples_pred = a_init_samples
            enkf_samples_post = a_init_samples
            
            if analyze_cdf_result:
                theta_post, params_post, theta_pred, params_pred = theta_init, params_init, theta_init, params_init
            
            # Initial Gaussian mixture parameters
            means_gsf = means_gm_init
            covs_gsf = covs_gm_init
            log_weights_gsf = log_weights_gm_init
            
            means_sp_gsf = means_gm_init
            covs_sp_gsf = covs_gm_init
            log_weights_sp_gsf = log_weights_gm_init
            
            means_pgm = means_gm_init
            covs_pgm = covs_gm_init
            log_weights_pgm = log_weights_gm_init
            
            means_pgm_2 = means_gm_init
            covs_pgm_2 = covs_gm_init
            log_weights_pgm_2 = log_weights_gm_init
        else:
            # Get particle samples from history
            samples_post = x_particle_posterior_history[time_index - 1]
            samples_pred = x_particle_prior_history[time_index - 1]
            enkf_samples_post = enkf_particle_posterior_history[time_index - 1]
            
            if analyze_cdf_result:
                # Get projection filter parameters
                theta_post, params_post, theta_pred, params_pred = _extract_parameters(
                    cdf_results, time_index, theta_init, params_init)
            
            # Get Gaussian mixture parameters
            means_gsf = gsf_results[0][time_index - 1]
            covs_gsf = gsf_results[1][time_index - 1]
            log_weights_gsf = gsf_results[2][time_index - 1]
            
            means_sp_gsf = sp_gsf_results[0][time_index - 1]
            covs_sp_gsf = sp_gsf_results[1][time_index - 1]
            log_weights_sp_gsf = sp_gsf_results[2][time_index - 1]

            # Handle PGM results - use NaN if empty
            if pgm_results is not None and len(pgm_results) > 0 and len(pgm_results[0]) > 0:
                means_pgm = pgm_results[0][time_index - 1]
                covs_pgm = pgm_results[1][time_index - 1]
                log_weights_pgm = jnp.log(pgm_results[2][time_index - 1])
            else:
                logger.warning("PGM results are empty, using NaN placeholders")
                num_clusters = sim_config.gauss_mixture_clusters
                means_pgm = jnp.full((num_clusters, dynamic.dim_states), jnp.nan)
                covs_pgm = jnp.full((num_clusters, dynamic.dim_states, dynamic.dim_states), jnp.nan)
                log_weights_pgm = jnp.full(num_clusters, jnp.nan)

            # Handle PGM-EM results - use NaN if empty
            if pgm_em_results is not None and len(pgm_em_results) > 0 and len(pgm_em_results[0]) > 0:
                means_pgm_2 = pgm_em_results[0][time_index - 1]
                covs_pgm_2 = pgm_em_results[1][time_index - 1]
                log_weights_pgm_2 = jnp.log(pgm_em_results[2][time_index - 1])
            else:
                logger.warning("PGM-EM results are empty, using NaN placeholders")
                num_clusters = sim_config.gauss_mixture_clusters
                means_pgm_2 = jnp.full((num_clusters, dynamic.dim_states), jnp.nan)
                covs_pgm_2 = jnp.full((num_clusters, dynamic.dim_states, dynamic.dim_states), jnp.nan)
                log_weights_pgm_2 = jnp.full(num_clusters, jnp.nan)

        # Calculate grid info from particles
        grid_limits_post_particle, post_bins, dxs_post = _get_particles_grid_info(samples_post, sim_config)
        grid_limits_pred_particle, pred_bins, dxs_pred = _get_particles_grid_info(samples_pred, sim_config)
        
        # Calculate particle densities
        density_particle_post = _calculate_particle_density(samples_post, post_bins)
        density_particle_pred = _calculate_particle_density(samples_pred, pred_bins)
        density_enkf_post = _calculate_particle_density(enkf_samples_post, post_bins)

        # Calculate natural statistics expectations
        if analyze_cdf_result:
            eta_history_pred.append(p.natural_statistics_expectation(theta_pred, params_pred))
            eta_history_post.append(p.natural_statistics_expectation(theta_post, params_post))

        eta_history_pred_particle.append(jnp.mean(p.natural_statistics(samples_pred), axis=0))
        eta_history_post_particle.append(jnp.mean(p.natural_statistics(samples_post), axis=0))
        eta_history_enkf.append(jnp.mean(p.natural_statistics(enkf_samples_post), axis=0))

        # Calculate densities for all filters
        if analyze_cdf_result:
            grid_pred, density_pred = p.get_density_values(grid_limits_pred_particle, theta_pred, num_points, params_pred)
            grid_post, density_post = p.get_density_values(grid_limits_post_particle, theta_post, num_points, params_post)
        
        density_gsf = dm.mix_gaussian_densities_on_grid(
            grid_limits_post_particle, sim_config.n_point_per_axis, 
            log_weights_gsf, means_gsf, covs_gsf
        )
        
        density_sp_gsf = dm.mix_gaussian_densities_on_grid(
            grid_limits_post_particle, sim_config.n_point_per_axis, 
            log_weights_sp_gsf, means_sp_gsf, covs_sp_gsf
        )
        
        density_pgm = dm.mix_gaussian_densities_on_grid(
            grid_limits_post_particle, sim_config.n_point_per_axis, 
            log_weights_pgm, means_pgm, covs_pgm
        )
        
        density_pgm_2 = dm.mix_gaussian_densities_on_grid(
            grid_limits_post_particle, sim_config.n_point_per_axis, 
            log_weights_pgm_2, means_pgm_2, covs_pgm_2
        )

        # Calculate divergence metrics
        if time_index < measurement_record.shape[0] and analyze_cdf_result:  # Only for valid measurement indices
            Z_theta_pred = p.log_expected_value(
                lambda _x: -dynamic.negative_log_likelihood(_x, measurement_record[time_index]), 
                theta_pred, params_pred
            )
            psi_theta_pred = p.log_partition(theta_pred, params_pred)
            
            D_half = bu.divergence_half(
                p, dynamic.negative_log_likelihood,
                theta_post, theta_pred, measurement_record[time_index], 
                params_post, psi_theta_pred, Z_theta_pred
            )
            H_pq = jnp.sqrt(1 - jnp.exp(-D_half / 2))
            D_half_hist.append(D_half)
            H_pq_hist.append(H_pq)

        # Calculate cross-entropy metrics
        if analyze_cdf_result:
            cross_entropy_hist_pred.append(p.cross_entropy(samples_pred, theta_pred, params_pred))
            cross_entropy_hist_post.append(p.cross_entropy(samples_post, theta_post, params_post))
        cross_entropy_hist_gsf.append(
            dm.gaussian_mixture_cross_entropy(samples_post, means_gsf, covs_gsf, log_weights_gsf))
        cross_entropy_hist_sp_gsf.append(
            dm.gaussian_mixture_cross_entropy(samples_post, means_sp_gsf, covs_sp_gsf, log_weights_sp_gsf))
        cross_entropy_hist_pgm.append(
            dm.gaussian_mixture_cross_entropy(samples_post, means_pgm, covs_pgm, log_weights_pgm))
        cross_entropy_hist_pgm_2.append(
            dm.gaussian_mixture_cross_entropy(samples_post, means_pgm_2, covs_pgm_2, log_weights_pgm_2))

        # Store density results
        if analyze_cdf_result:
            density_hist_post.append(density_post)
            density_hist_pred.append(density_pred)
        density_particle_hist_pred.append(density_particle_pred)
        density_particle_hist_post.append(density_particle_post)
        density_sp_gsf_hist_post.append(density_sp_gsf)
        density_gsf_hist_post.append(density_gsf)
        density_pgm_hist_post.append(density_pgm)
        density_pgm_2_hist_post.append(density_pgm_2)
        density_enkf_hist_post.append(density_enkf_post)

        # Calculate Hellinger distances
        if analyze_cdf_result:
            hell_dist_hist_post.append(dm.hellinger_distance(density_particle_post, density_post, dxs_post))
            hell_dist_hist_pred.append(dm.hellinger_distance(density_particle_pred, density_pred, dxs_pred))
        hell_dist_hist_gsf.append(dm.hellinger_distance(density_particle_post, density_gsf, dxs_post))
        hell_dist_hist_sp_gsf.append(dm.hellinger_distance(density_particle_post, density_sp_gsf, dxs_post))
        hell_dist_hist_pgm.append(dm.hellinger_distance(density_particle_post, density_pgm, dxs_post))
        hell_dist_hist_pgm_2.append(dm.hellinger_distance(density_particle_post, density_pgm_2, dxs_post))
        hell_dist_hist_enkf.append(dm.hellinger_distance(density_particle_post, density_enkf_post, dxs_post))

        # Store grid info
        grid_limits_hist_post.append(grid_limits_post_particle)
        grid_limits_hist_pred.append(grid_limits_pred_particle)
        if analyze_cdf_result:
            grid_hist_post.append(grid_post)


    # Stack all results
    results = {
        "D_half_hist": jnp.stack(D_half_hist) if D_half_hist else None,
        "H_pq_hist": jnp.stack(H_pq_hist) if H_pq_hist else None,
        "eta_history_pred": jnp.stack(eta_history_pred) if eta_history_pred else None,
        "eta_history_post": jnp.stack(eta_history_post) if eta_history_post else None,
        "eta_history_pred_particle": jnp.stack(eta_history_pred_particle) if eta_history_pred_particle else None,
        "eta_history_post_particle": jnp.stack(eta_history_post_particle) if eta_history_post_particle else None,
        "eta_history_enkf": jnp.stack(eta_history_enkf),
        "cross_entropy_hist_pred": jnp.stack(cross_entropy_hist_pred) if cross_entropy_hist_pred else None,
        "cross_entropy_hist_post": jnp.stack(cross_entropy_hist_post) if cross_entropy_hist_post else None,
        "cross_entropy_hist_gsf": jnp.stack(cross_entropy_hist_gsf),
        "cross_entropy_hist_sp_gsf": jnp.stack(cross_entropy_hist_sp_gsf),
        "cross_entropy_hist_pgm": jnp.stack(cross_entropy_hist_pgm),
        "cross_entropy_hist_pgm_2": jnp.stack(cross_entropy_hist_pgm_2),
        "density_hist_pred": jnp.stack(density_hist_pred) if density_hist_pred else None,
        "density_hist_post": jnp.stack(density_hist_post) if density_hist_post else None,
        "density_particle_hist_pred": jnp.stack(density_particle_hist_pred),
        "density_particle_hist_post": jnp.stack(density_particle_hist_post),
        "density_sp_gsf_hist_post": jnp.stack(density_sp_gsf_hist_post),
        "density_gsf_hist_post": jnp.stack(density_gsf_hist_post),
        "density_pgm_hist_post": jnp.stack(density_pgm_hist_post),
        "density_pgm_2_hist_post": jnp.stack(density_pgm_2_hist_post),
        "density_enkf_hist_post": jnp.stack(density_enkf_hist_post),
        "hell_dist_hist_post": jnp.stack(hell_dist_hist_post) if hell_dist_hist_post else None,
        "hell_dist_hist_pred": jnp.stack(hell_dist_hist_pred) if hell_dist_hist_pred else None,
        "hell_dist_hist_gsf": jnp.stack(hell_dist_hist_gsf),
        "hell_dist_hist_sp_gsf": jnp.stack(hell_dist_hist_sp_gsf),
        "hell_dist_hist_pgm": jnp.stack(hell_dist_hist_pgm),
        "hell_dist_hist_pgm_2": jnp.stack(hell_dist_hist_pgm_2),
        "hell_dist_hist_enkf": jnp.stack(hell_dist_hist_enkf),
        "grid_limits_hist_post": jnp.stack(grid_limits_hist_post),
        "grid_limits_hist_pred": jnp.stack(grid_limits_hist_pred),
        "grid_hist_post": jnp.stack(grid_hist_post) if grid_hist_post else None,
    }


    # Check for NaN values in density
    if analyze_cdf_result:
        if jnp.any(jnp.isnan(results["density_hist_post"])):
            logger.warning("There is a nan in density_hist_post!")

    return results

def _post_analysis_full_for_high_dim(dynamic: DynamicalSystem,
                                sim_config: SimulationConfig,
                                par_config: ParticleFilterConfig,
                                p: NDExponentialFamily,
                                init_samples: Array,
                                measurement_record: Array,
                                n_devices: int,
                                means_gm_init: Array,
                                covs_gm_init: Array,
                                log_weights_gm_init: Array,
                                theta_init: Array,
                                params_init: tuple[Array, Array, Float],
                                x_particle_posterior_history: Array,
                                x_particle_prior_history: Array,
                                enkf_particle_posterior_history: Array,
                                cdf_results: Any,
                                gsf_results: Any,
                                sp_gsf_results: Any,
                                pgm_results: Any,
                                pgm_em_results: Any,
                                x_meas: Array,
                                theta_indices_for_mean: Array,
                                logger: logging.Logger,
                                save_samples: bool = False,
                                n_sw_projections: int = 100,
                                n_ef_samples: int = 10000) -> Dict[str, Array]:
    """Perform post-analysis for high-dimensional systems (dim > 2).

    Uses sample-based Sliced Wasserstein-1 distance instead of grid-based Hellinger
    distance, which is infeasible for dimensions greater than 2.

    FULLY VECTORIZED VERSION: Uses jax.vmap to process all timesteps in a single
    JIT compilation, avoiding the costly retracing that occurs with Python loops.

    Key optimizations:
    1. Pre-stack all data arrays across timesteps before computation
    2. Use jax.vmap to vectorize over timesteps (seeds as JAX array, not Python int)
    3. Single JIT compilation per vmapped function (not per timestep)

    Parameters
    ----------
    dynamic : DynamicalSystem
        Dynamical system model.
    sim_config : SimulationConfig
        Simulation configuration.
    par_config : ParticleFilterConfig
        Particle filter configuration.
    p : NDExponentialFamily
        Exponential family distribution.
    init_samples : Array
        Initial samples.
    measurement_record : Array
        Measurement data.
    n_devices : int
        Number of devices.
    means_gm_init, covs_gm_init, log_weights_gm_init : Arrays
        Initial Gaussian mixture parameters.
    theta_init : Array
        Initial natural parameters.
    params_init : tuple
        Initial bijection parameters.
    x_particle_posterior_history, x_particle_prior_history : Arrays
        Particle filter sample histories.
    enkf_particle_posterior_history : Array
        EnKF sample history.
    cdf_results : Any
        Projection filter results.
    gsf_results, sp_gsf_results, pgm_results, pgm_em_results : Any
        Various filter results.
    x_meas : Array
        Measurement states.
    logger : logging.Logger
        Logger for messages.
    save_samples : bool
        If True, include raw samples in results dict.
    n_sw_projections : int
        Number of random projections for Sliced Wasserstein distance.
    n_ef_samples : int
        Number of samples to draw from exponential family for SW computation.

    Returns
    -------
    Dict[str, Array]
        Analysis results with Sliced Wasserstein distances instead of Hellinger.
    """
    # Check if the cdf_results is None
    analyze_cdf_result = cdf_results is not None
    if not analyze_cdf_result:
        logger.warning("CDF results are None, skipping projection filter analysis.")

    # Reshape initial samples
    n_particles = n_devices * par_config.n_particle_per_device
    a_init_samples = init_samples.reshape((n_particles, dynamic.dim_states))

    n_timesteps = measurement_record.shape[0] + 1
    num_clusters = sim_config.gauss_mixture_clusters

    logger.info(f"Post-analysis (high-dim): Starting with {n_timesteps} timesteps, {n_particles} particles")

    # =========================================================================
    # STEP 1: Pre-stack ALL data arrays across timesteps
    # This is critical for vectorization - no Python loops over timesteps
    # =========================================================================

    # Stack particle samples: shape (n_timesteps, n_particles, dim)
    all_post_samples = jnp.concatenate([a_init_samples[None, :, :], x_particle_posterior_history], axis=0)
    all_prior_samples = jnp.concatenate([a_init_samples[None, :, :], x_particle_prior_history], axis=0)
    all_enkf_samples = jnp.concatenate([a_init_samples[None, :, :], enkf_particle_posterior_history], axis=0)

    # Subsample particle samples for SW distance computation (memory efficiency)
    # Cross-entropy uses full arrays (O(n) and fast), but SW distance is memory-intensive
    max_samples_for_metric = n_ef_samples  # Use same number as EF samples
    if n_particles > max_samples_for_metric:
        logger.info(f"Post-analysis (high-dim): Subsampling particles from {n_particles} to {max_samples_for_metric} for SW distance")
        subsample_key = jrandom.PRNGKey(sim_config.seed + 2000)
        subsample_indices = jrandom.choice(subsample_key, n_particles, shape=(max_samples_for_metric,), replace=False)
        all_post_samples_sub = all_post_samples[:, subsample_indices, :]
        all_prior_samples_sub = all_prior_samples[:, subsample_indices, :]
        all_enkf_samples_sub = all_enkf_samples[:, subsample_indices, :]
    else:
        all_post_samples_sub = all_post_samples
        all_prior_samples_sub = all_prior_samples
        all_enkf_samples_sub = all_enkf_samples

    # Stack GSF parameters: shape (n_timesteps, n_clusters, ...)
    all_means_gsf = jnp.concatenate([means_gm_init[None], gsf_results[0]], axis=0)
    all_covs_gsf = jnp.concatenate([covs_gm_init[None], gsf_results[1]], axis=0)
    all_logweights_gsf = jnp.concatenate([log_weights_gm_init[None], gsf_results[2]], axis=0)

    # Stack SP-GSF parameters
    all_means_sp_gsf = jnp.concatenate([means_gm_init[None], sp_gsf_results[0]], axis=0)
    all_covs_sp_gsf = jnp.concatenate([covs_gm_init[None], sp_gsf_results[1]], axis=0)
    all_logweights_sp_gsf = jnp.concatenate([log_weights_gm_init[None], sp_gsf_results[2]], axis=0)

    # Stack PGM parameters (with NaN handling)
    pgm_available = pgm_results is not None and len(pgm_results) > 0 and len(pgm_results[0]) > 0
    if pgm_available:
        # PGM stores weights (not log_weights), so convert
        all_means_pgm = jnp.concatenate([means_gm_init[None], pgm_results[0]], axis=0)
        all_covs_pgm = jnp.concatenate([covs_gm_init[None], pgm_results[1]], axis=0)
        all_logweights_pgm = jnp.concatenate([log_weights_gm_init[None], jnp.log(pgm_results[2])], axis=0)
    else:
        logger.warning("PGM results are empty, using NaN placeholders for all timesteps")
        nan_means = jnp.full((n_timesteps, num_clusters, dynamic.dim_states), jnp.nan)
        nan_covs = jnp.full((n_timesteps, num_clusters, dynamic.dim_states, dynamic.dim_states), jnp.nan)
        nan_logweights = jnp.full((n_timesteps, num_clusters), jnp.nan)
        all_means_pgm, all_covs_pgm, all_logweights_pgm = nan_means, nan_covs, nan_logweights

    # Stack PGM-EM parameters (with NaN handling)
    pgm_em_available = pgm_em_results is not None and len(pgm_em_results) > 0 and len(pgm_em_results[0]) > 0
    if pgm_em_available:
        all_means_pgm_2 = jnp.concatenate([means_gm_init[None], pgm_em_results[0]], axis=0)
        all_covs_pgm_2 = jnp.concatenate([covs_gm_init[None], pgm_em_results[1]], axis=0)
        all_logweights_pgm_2 = jnp.concatenate([log_weights_gm_init[None], jnp.log(pgm_em_results[2])], axis=0)
    else:
        logger.warning("PGM-EM results are empty, using NaN placeholders for all timesteps")
        if not pgm_available:  # Reuse nan arrays if already created
            all_means_pgm_2, all_covs_pgm_2, all_logweights_pgm_2 = nan_means, nan_covs, nan_logweights
        else:
            nan_means = jnp.full((n_timesteps, num_clusters, dynamic.dim_states), jnp.nan)
            nan_covs = jnp.full((n_timesteps, num_clusters, dynamic.dim_states, dynamic.dim_states), jnp.nan)
            nan_logweights = jnp.full((n_timesteps, num_clusters), jnp.nan)
            all_means_pgm_2, all_covs_pgm_2, all_logweights_pgm_2 = nan_means, nan_covs, nan_logweights

    # Stack CDF results if available
    if analyze_cdf_result:
        all_theta_post = jnp.vstack([theta_init[None, :], cdf_results[0]])
        all_mu_post = jnp.vstack([params_init[0][None, :], cdf_results[1][0]])
        all_chol_post = jnp.concatenate([params_init[1][None, :, :], cdf_results[1][1]], axis=0)

        # Handle different cdf_results structures for prediction parameters
        # Check if cdf_results[3] has multiple prediction steps (nested tuple) or single step (array)
        if isinstance(cdf_results[3], tuple) and len(cdf_results[3]) > 1:
            # Multiple prediction steps: cdf_results[3] = (theta_history, mu_history, chol_history)
            # Take last prediction step for each timestep
            all_theta_pred = jnp.vstack([theta_init[None, :], cdf_results[3][0][:, -1, :]])
            all_mu_pred = jnp.vstack([params_init[0][None, :], cdf_results[3][1][:, -1, :]])
            all_chol_pred = jnp.concatenate([params_init[1][None, :, :], cdf_results[3][2][:, -1, :, :]], axis=0)
        else:
            # Single prediction step per measurement
            all_theta_pred = jnp.vstack([theta_init[None, :], cdf_results[3]])
            all_mu_pred = jnp.vstack([params_init[0][None, :], cdf_results[4][0]])
            all_chol_pred = jnp.concatenate([params_init[1][None, :, :], cdf_results[4][1]], axis=0)

    logger.info("Post-analysis (high-dim): Step 1/5 - Data arrays stacked")

    # =========================================================================
    # STEP 2: Define vectorized computation functions using jax.vmap
    # Key: seeds are JAX arrays, not Python ints - avoids retracing!
    # =========================================================================

    # Vectorized eta computation from samples
    @filter_jit
    def compute_all_eta_from_samples(all_samples):
        """Compute eta for all timesteps at once. Shape: (n_timesteps, n_particles, dim) -> (n_timesteps, eta_dim)"""
        return jax.vmap(lambda s: jnp.mean(p.natural_statistics(s), axis=0))(all_samples)

    # Vectorized SW distance computation - seeds as JAX array!
    @filter_jit
    def compute_all_sw_distances(ref_batch, target_batch, seeds_array):
        """Compute SW distances for all timesteps at once."""
        def single_sw(ref, target, seed):
            return metrics.sliced_wasserstein_distance_parallel(
                ref, target, n_projections=n_sw_projections, seed=seed)
        return jax.vmap(single_sw)(ref_batch, target_batch, seeds_array)

    # Vectorized GM cross-entropy
    @filter_jit
    def compute_all_gm_cross_entropy(all_samples, all_means, all_covs, all_logweights):
        """Compute GM cross-entropy for all timesteps at once."""
        def single_ce(samples, means, covs, logweights):
            return dm.gaussian_mixture_cross_entropy(samples, means, covs, logweights)
        return jax.vmap(single_ce)(all_samples, all_means, all_covs, all_logweights)

    # Vectorized GM sampling
    @filter_jit
    def sample_all_gm(all_means, all_covs, all_weights, keys):
        """Sample from GM for all timesteps at once."""
        def single_sample(means, covs, weights, key):
            return dm.sample_gaussian_mixture(means, covs, weights, n_ef_samples, key)
        return jax.vmap(single_sample)(all_means, all_covs, all_weights, keys)

    # CDF-specific functions
    if analyze_cdf_result:
        @filter_jit
        def compute_all_eta_ef(all_theta, all_mu, all_chol):
            """Compute eta from exponential family for all timesteps."""
            def single_eta(theta, mu, chol):
                return p.natural_statistics_expectation(theta, (mu, chol, 1.))
            return jax.vmap(single_eta)(all_theta, all_mu, all_chol)

        @filter_jit
        def compute_all_cross_entropy_ef(all_samples, all_theta, all_mu, all_chol):
            """Compute cross-entropy for all timesteps."""
            def single_ce(samples, theta, mu, chol):
                return p.cross_entropy(samples, theta, (mu, chol, 1.))
            return jax.vmap(single_ce)(all_samples, all_theta, all_mu, all_chol)

        @filter_jit
        def sample_all_ef(all_theta, all_mu, all_chol, keys):
            """Sample from exponential family for all timesteps."""
            def single_sample(theta, mu, chol, key):
                return p.sample(
                    (n_ef_samples, 1), theta, theta_indices_for_mean, (mu, chol, 1.), key
                ).squeeze()
            return jax.vmap(single_sample)(all_theta, all_mu, all_chol, keys)

    logger.info("Post-analysis (high-dim): Step 2/5 - Vectorized functions defined")

    # =========================================================================
    # STEP 3: Generate all PRNG keys upfront
    # =========================================================================
    base_key = jrandom.PRNGKey(sim_config.seed + 1000)
    # Seeds for SW distance (as JAX array - this is the key fix!)
    seeds = jnp.arange(n_timesteps)

    # Keys for sampling (one per timestep per method)
    all_keys = jrandom.split(base_key, n_timesteps * 7).reshape(7, n_timesteps, 2)
    keys_gsf = all_keys[0]
    keys_sp_gsf = all_keys[1]
    keys_pgm = all_keys[2]
    keys_pgm_2 = all_keys[3]
    keys_ef_post = all_keys[4]
    keys_ef_pred = all_keys[5]

    logger.info("Post-analysis (high-dim): Step 3/5 - PRNG keys generated")

    # =========================================================================
    # STEP 4: Execute all computations in vectorized fashion
    # Each call is ONE JIT compilation, processing all timesteps
    # =========================================================================

    # Compute eta from particle samples (3 calls, each processes all timesteps)
    logger.info("Post-analysis (high-dim): Step 4/5 - Computing eta from particle samples...")
    eta_history_post_particle = compute_all_eta_from_samples(all_post_samples)
    eta_history_pred_particle = compute_all_eta_from_samples(all_prior_samples)
    eta_history_enkf = compute_all_eta_from_samples(all_enkf_samples)

    # Compute cross-entropy for GM methods
    logger.info("Post-analysis (high-dim): Step 4/5 - Computing cross-entropy for GM methods...")
    cross_entropy_hist_gsf = compute_all_gm_cross_entropy(
        all_post_samples, all_means_gsf, all_covs_gsf, all_logweights_gsf)
    cross_entropy_hist_sp_gsf = compute_all_gm_cross_entropy(
        all_post_samples, all_means_sp_gsf, all_covs_sp_gsf, all_logweights_sp_gsf)
    cross_entropy_hist_pgm = compute_all_gm_cross_entropy(
        all_post_samples, all_means_pgm, all_covs_pgm, all_logweights_pgm)
    cross_entropy_hist_pgm_2 = compute_all_gm_cross_entropy(
        all_post_samples, all_means_pgm_2, all_covs_pgm_2, all_logweights_pgm_2)

    # Sample from all GMs (one call per method, all timesteps)
    logger.info("Post-analysis (high-dim): Step 4/5 - Sampling from Gaussian mixtures...")
    all_gsf_samples = sample_all_gm(all_means_gsf, all_covs_gsf, jnp.exp(all_logweights_gsf), keys_gsf)
    all_sp_gsf_samples = sample_all_gm(all_means_sp_gsf, all_covs_sp_gsf, jnp.exp(all_logweights_sp_gsf), keys_sp_gsf)
    all_pgm_samples = sample_all_gm(all_means_pgm, all_covs_pgm, jnp.exp(all_logweights_pgm), keys_pgm)
    all_pgm_2_samples = sample_all_gm(all_means_pgm_2, all_covs_pgm_2, jnp.exp(all_logweights_pgm_2), keys_pgm_2)

    # Compute SW distances (one call per comparison, all timesteps)
    # Use subsampled arrays for memory efficiency
    logger.info("Post-analysis (high-dim): Step 4/5 - Computing SW distances for benchmark methods...")
    sw1_dist_hist_gsf = compute_all_sw_distances(all_post_samples_sub, all_gsf_samples, seeds)
    sw1_dist_hist_sp_gsf = compute_all_sw_distances(all_post_samples_sub, all_sp_gsf_samples, seeds)
    sw1_dist_hist_pgm = compute_all_sw_distances(all_post_samples_sub, all_pgm_samples, seeds)
    sw1_dist_hist_pgm_2 = compute_all_sw_distances(all_post_samples_sub, all_pgm_2_samples, seeds)
    sw1_dist_hist_enkf = compute_all_sw_distances(all_post_samples_sub, all_enkf_samples_sub, seeds)

    # CDF-specific computations
    if analyze_cdf_result:
        logger.info("Post-analysis (high-dim): Step 4/5 - Computing projection filter metrics...")
        # Eta from exponential family
        eta_history_post = compute_all_eta_ef(all_theta_post, all_mu_post, all_chol_post)
        eta_history_pred = compute_all_eta_ef(all_theta_pred, all_mu_pred, all_chol_pred)

        # Cross-entropy
        cross_entropy_hist_post = compute_all_cross_entropy_ef(
            all_post_samples, all_theta_post, all_mu_post, all_chol_post)
        cross_entropy_hist_pred = compute_all_cross_entropy_ef(
            all_prior_samples, all_theta_pred, all_mu_pred, all_chol_pred)

        # Sample from exponential family
        all_ef_samples_post = sample_all_ef(all_theta_post, all_mu_post, all_chol_post, keys_ef_post)
        all_ef_samples_pred = sample_all_ef(all_theta_pred, all_mu_pred, all_chol_pred, keys_ef_pred)

        # SW distances for projection filter (use subsampled arrays for memory efficiency)
        sw1_dist_hist_post = compute_all_sw_distances(all_post_samples_sub, all_ef_samples_post, seeds)
        sw1_dist_hist_pred = compute_all_sw_distances(all_prior_samples_sub, all_ef_samples_pred, seeds + 1000)

    # =========================================================================
    # STEP 5: Build results dictionary
    # =========================================================================
    logger.info("Post-analysis (high-dim): Step 5/5 - Building results dictionary...")
    results = {
        "eta_history_pred": eta_history_pred if analyze_cdf_result else None,
        "eta_history_post": eta_history_post if analyze_cdf_result else None,
        "eta_history_pred_particle": eta_history_pred_particle,
        "eta_history_post_particle": eta_history_post_particle,
        "eta_history_enkf": eta_history_enkf,
        "cross_entropy_hist_pred": cross_entropy_hist_pred if analyze_cdf_result else None,
        "cross_entropy_hist_post": cross_entropy_hist_post if analyze_cdf_result else None,
        "cross_entropy_hist_gsf": cross_entropy_hist_gsf,
        "cross_entropy_hist_sp_gsf": cross_entropy_hist_sp_gsf,
        "cross_entropy_hist_pgm": cross_entropy_hist_pgm,
        "cross_entropy_hist_pgm_2": cross_entropy_hist_pgm_2,
        "sw1_dist_hist_post": sw1_dist_hist_post if analyze_cdf_result else None,
        "sw1_dist_hist_pred": sw1_dist_hist_pred if analyze_cdf_result else None,
        "sw1_dist_hist_gsf": sw1_dist_hist_gsf,
        "sw1_dist_hist_sp_gsf": sw1_dist_hist_sp_gsf,
        "sw1_dist_hist_pgm": sw1_dist_hist_pgm,
        "sw1_dist_hist_pgm_2": sw1_dist_hist_pgm_2,
        "sw1_dist_hist_enkf": sw1_dist_hist_enkf,
    }

    # Add samples if requested
    if save_samples:
        results["particle_posterior_samples"] = all_post_samples
        results["particle_prior_samples"] = all_prior_samples
        results["enkf_posterior_samples"] = all_enkf_samples
        if analyze_cdf_result:
            results["proj_filter_samples_post"] = all_ef_samples_post
            results["proj_filter_samples_pred"] = all_ef_samples_pred

    return results