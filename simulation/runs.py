import logging
from datetime import datetime, timedelta
from typing import Callable

import jax
import jax.numpy as jnp
import sympy as sp
import jax.random as jrandom
import cd_filtering.cd_proj_conjugate
import cd_filtering.flows as fl
import symbolic.n_d as nds
from exponential_family.n_d_ef import NDExponentialFamily
from exponential_family.n_d_ef_spg import NDExponentialFamilySPG, gauss_bijection_original
from other_filter.cd_ekf import cd_gsf, create_ekf_ode_for_cost_analysis, create_gsf_update_for_cost_analysis
from other_filter.cd_enkf import cd_enkf, create_enkf_drift_for_cost_analysis, create_enkf_update_for_cost_analysis
from other_filter.cd_pgm import cd_pgm_optimized as cd_pgm, create_pgm_drift_for_cost_analysis, create_pgm_update_for_cost_analysis
from other_filter.cd_pgm_em import cd_pgm as cd_pgm_em, create_pgm_em_drift_for_cost_analysis, \
    create_pgm_em_update_for_cost_analysis
from other_filter.cd_sp_kf import cd_sp_gsf, create_spkf_ode_for_cost_analysis, create_spgsf_update_for_cost_analysis
from other_filter.particlefilter_cont_discrete import ContinuousDiscreteParticleFilter, \
    create_particle_drift_for_cost_analysis, create_particle_update_for_cost_analysis
from sigma_points.gauss_hermite import GaussHermiteSigmaPoints
from simulation.configs import SimulationConfig, ProjectionFilterConfig, ParticleFilterConfig, \
    EnKFConfig
from simulation.dynamical_system import DynamicalSystem
from symbolic.sympy_to_jax import sympy_matrix_to_jax
from cd_filtering.cd_proj_conjugate import get_theta_ell, cd_conj_proj_filter_chol,\
    create_bayesian_update_for_cost_analysis
from cd_filtering.flows import create_fokker_planck_flow_cholesky_for_cost_analysis
from jaxtyping import Array, Float


def compute_corrected_flops_with_ode(cost_analysis: dict, n_meas: int, fp_sol_stats: dict) -> dict:
    """
    Compute corrected FLOPS accounting for n_meas and actual ODE steps.

    JAX's cost_analysis() returns FLOPS for a single scan iteration and
    assumes 1 ODE step. This function corrects for actual iteration counts.

    Note: fp_sol_stats['num_steps'] is an array of shape (n_meas,) where each
    element is the number of ODE steps taken for that measurement. The steps
    can vary between measurements due to adaptive step size control.

    Parameters
    ----------
    cost_analysis : dict
        The raw cost analysis from compiled.cost_analysis()[0].
    n_meas : int
        Number of measurements (scan iterations).
    fp_sol_stats : dict
        Statistics from diffrax solution (fp_sol.stats) containing 'num_steps'.

    Returns
    -------
    dict
        Updated cost analysis with corrected FLOPS and additional metadata.
    """
    raw_flops = cost_analysis.get("flops", float('nan'))
    ode_steps_per_meas = fp_sol_stats['num_steps']  # Array of shape (n_meas,)
    total_ode_steps = int(jnp.sum(ode_steps_per_meas))
    mean_ode_steps = float(jnp.mean(ode_steps_per_meas))
    min_ode_steps = int(jnp.min(ode_steps_per_meas))
    max_ode_steps = int(jnp.max(ode_steps_per_meas))

    # Corrected FLOPS = raw_flops_per_step * total_ode_steps
    # Since raw_flops is for 1 scan iteration with 1 ODE step
    corrected_flops = raw_flops * total_ode_steps

    return {
        **cost_analysis,
        "raw_flops": raw_flops,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_ode_steps,
        "mean_ode_steps_per_meas": mean_ode_steps,
        "min_ode_steps_per_meas": min_ode_steps,
        "max_ode_steps_per_meas": max_ode_steps,
    }


def compute_decomposed_flops_fixed_steps(
    flow_derivative_fn: Callable,
    sample_states: tuple,
    bayesian_update_fn: Callable,
    sample_params: tuple,
    sample_meas: Array,
    n_meas: int,
    ode_steps_per_meas: int
) -> dict:
    """
    Compute FLOPS with decomposition for fixed-step ODE solvers (RK4).

    For filters using fixed-step RK4 solvers (SOS, constrained projection filters),
    we know the exact number of ODE steps per measurement a priori.

    Parameters
    ----------
    flow_derivative_fn : Callable
        JIT-compiled flow derivative function.
    sample_states : tuple
        Sample states with correct shapes: (theta, mu_params, cov_params).
    bayesian_update_fn : Callable
        JIT-compiled Bayesian update function.
    sample_params : tuple
        Sample params tuple with correct shapes.
    sample_meas : Array
        Single measurement array with correct shape.
    n_meas : int
        Number of measurements (scan iterations).
    ode_steps_per_meas : int
        Number of ODE steps per measurement (fixed for RK4 solvers).

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information.
    """
    # Compile flow derivative and get FLOPS for one ODE step
    compiled_flow = flow_derivative_fn.lower(0.0, sample_states, None).compile()
    flops_per_ode_step = compiled_flow.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile Bayesian update and get FLOPS
    theta_sample = sample_states[0]
    compiled_bayesian = bayesian_update_fn.lower(theta_sample, sample_params, sample_meas).compile()
    flops_bayesian = compiled_bayesian.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compute totals (fixed step size, so all measurements have the same ODE step count)
    total_ode_steps = n_meas * ode_steps_per_meas

    flops_ode_total = flops_per_ode_step * total_ode_steps
    flops_bayesian_total = flops_bayesian * n_meas
    corrected_flops = flops_ode_total + flops_bayesian_total

    return {
        "flops_per_ode_step": flops_per_ode_step,
        "flops_bayesian_per_meas": flops_bayesian,
        "flops_ode_total": flops_ode_total,
        "flops_bayesian_total": flops_bayesian_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_ode_steps,
        "ode_steps_per_meas": ode_steps_per_meas,
        "ode_to_bayesian_ratio": flops_ode_total / flops_bayesian_total if flops_bayesian_total > 0 else float('inf'),
    }


def compute_decomposed_flops(
    flow_derivative_fn: Callable,
    sample_states: tuple,
    bayesian_update_fn: Callable,
    sample_params: tuple,
    sample_meas: Array,
    n_meas: int,
    fp_sol_stats: dict
) -> dict:
    """
    Compute FLOPS with decomposition into ODE and Bayesian update components.

    This function separately measures FLOPS for the ODE flow derivative and the
    Bayesian update, then combines them correctly using the formula:
        corrected_flops = (flops_per_ode_step * total_ode_steps) + (flops_bayesian * n_meas)

    This is more accurate than the simple formula (raw_flops * total_ode_steps)
    because the Bayesian update portion does NOT scale with ODE steps.

    Note: For cost_analysis(), only array shapes matter, not actual values.
    We use theta_init as a stand-in for theta_pred since they have identical shapes.

    Parameters
    ----------
    flow_derivative_fn : Callable
        JIT-compiled flow derivative function (from create_fokker_planck_flow_*_for_cost_analysis).
    sample_states : tuple
        Sample states with correct shapes: (theta, mu_params, cov_params).
        theta_init can be used since it has the same shape as theta_pred.
    bayesian_update_fn : Callable
        JIT-compiled Bayesian update function (from create_bayesian_update_for_cost_analysis).
    sample_params : tuple
        Sample params tuple with correct shapes.
    sample_meas : Array
        Single measurement array with correct shape.
    n_meas : int
        Number of measurements (scan iterations).
    fp_sol_stats : dict
        Statistics from diffrax solution containing 'num_steps' array.

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information:
        - flops_per_ode_step: FLOPS for one ODE derivative evaluation
        - flops_bayesian_per_meas: FLOPS for one Bayesian update
        - flops_ode_total: Total ODE FLOPS
        - flops_bayesian_total: Total Bayesian update FLOPS
        - corrected_flops: Sum of ODE and Bayesian FLOPS
        - n_meas: Number of measurements
        - total_ode_steps: Total ODE steps across all measurements
        - mean_ode_steps_per_meas: Average ODE steps per measurement
    """
    # Compile flow derivative and get FLOPS for one ODE step
    compiled_flow = flow_derivative_fn.lower(0.0, sample_states, None).compile()
    flops_per_ode_step = compiled_flow.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile Bayesian update and get FLOPS
    # theta_pred has same shape as theta_init, so we use sample_states[0] as theta_pred
    theta_sample = sample_states[0]
    compiled_bayesian = bayesian_update_fn.lower(theta_sample, sample_params, sample_meas).compile()
    flops_bayesian = compiled_bayesian.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Get ODE step counts
    ode_steps_per_meas = fp_sol_stats['num_steps']
    total_ode_steps = int(jnp.sum(ode_steps_per_meas))
    mean_ode_steps = float(jnp.mean(ode_steps_per_meas))
    min_ode_steps = int(jnp.min(ode_steps_per_meas))
    max_ode_steps = int(jnp.max(ode_steps_per_meas))

    # Compute corrected FLOPS with decomposition
    flops_ode_total = flops_per_ode_step * total_ode_steps
    flops_bayesian_total = flops_bayesian * n_meas
    corrected_flops = flops_ode_total + flops_bayesian_total

    return {
        "flops_per_ode_step": flops_per_ode_step,
        "flops_bayesian_per_meas": flops_bayesian,
        "flops_ode_total": flops_ode_total,
        "flops_bayesian_total": flops_bayesian_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_ode_steps,
        "mean_ode_steps_per_meas": mean_ode_steps,
        "min_ode_steps_per_meas": min_ode_steps,
        "max_ode_steps_per_meas": max_ode_steps,
        "ode_to_bayesian_ratio": flops_ode_total / flops_bayesian_total if flops_bayesian_total > 0 else float('inf'),
    }


def compute_decomposed_flops_gsf(
    ode_derivative_fn: Callable,
    sample_ode_states: tuple,
    update_fn: Callable,
    sample_mus: Array,
    sample_Ps: Array,
    sample_log_weights: Array,
    sample_meas: Array,
    n_meas: int,
    sol_stats: dict
) -> dict:
    """
    Compute FLOPS with decomposition for GSF/SP-GSF filters.

    Parameters
    ----------
    ode_derivative_fn : Callable
        JIT-compiled ODE derivative function.
    sample_ode_states : tuple
        Sample ODE states: (mus, Ps).
    update_fn : Callable
        JIT-compiled Bayesian update function.
    sample_mus : Array
        Sample means array.
    sample_Ps : Array
        Sample covariances array.
    sample_log_weights : Array
        Sample log weights array.
    sample_meas : Array
        Single measurement array.
    n_meas : int
        Number of measurements.
    sol_stats : dict
        Statistics from diffrax solution containing 'num_steps'.

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information.
    """
    # Compile ODE derivative and get FLOPS for one ODE step
    compiled_ode = ode_derivative_fn.lower(0.0, sample_ode_states, None).compile()
    flops_per_ode_step = compiled_ode.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile Bayesian update and get FLOPS
    compiled_update = update_fn.lower(sample_mus, sample_Ps, sample_log_weights, sample_meas).compile()
    flops_update = compiled_update.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Get ODE step counts
    ode_steps_per_meas = sol_stats['num_steps']
    total_ode_steps = int(jnp.sum(ode_steps_per_meas))
    mean_ode_steps = float(jnp.mean(ode_steps_per_meas))
    min_ode_steps = int(jnp.min(ode_steps_per_meas))
    max_ode_steps = int(jnp.max(ode_steps_per_meas))

    # Compute corrected FLOPS with decomposition
    flops_ode_total = flops_per_ode_step * total_ode_steps
    flops_update_total = flops_update * n_meas
    corrected_flops = flops_ode_total + flops_update_total

    return {
        "flops_per_ode_step": flops_per_ode_step,
        "flops_bayesian_per_meas": flops_update,
        "flops_ode_total": flops_ode_total,
        "flops_bayesian_total": flops_update_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_ode_steps,
        "mean_ode_steps_per_meas": mean_ode_steps,
        "min_ode_steps_per_meas": min_ode_steps,
        "max_ode_steps_per_meas": max_ode_steps,
        "ode_to_bayesian_ratio": flops_ode_total / flops_update_total if flops_update_total > 0 else float('inf'),
    }


def compute_decomposed_flops_ensemble(
    drift_fn: Callable,
    sample_samples: Array,
    update_fn: Callable,
    sample_meas: Array,
    sample_meas_noise: Array,
    n_meas: int,
    sol_stats: dict
) -> dict:
    """
    Compute FLOPS with decomposition for ensemble filters (EnKF, Particle Filter, PGM).

    Parameters
    ----------
    drift_fn : Callable
        JIT-compiled drift function for SDE.
    sample_samples : Array
        Sample ensemble array with correct shape.
    update_fn : Callable
        JIT-compiled Bayesian update function.
    sample_meas : Array
        Single measurement array.
    sample_meas_noise : Array
        Sample measurement noise array for perturbed observations.
    n_meas : int
        Number of measurements.
    sol_stats : dict
        Statistics from diffrax solution containing 'num_steps'.

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information.
    """
    # Compile drift and get FLOPS for one SDE step
    compiled_drift = drift_fn.lower(0.0, sample_samples, None).compile()
    flops_per_sde_step = compiled_drift.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile Bayesian update and get FLOPS
    compiled_update = update_fn.lower(sample_samples, sample_meas, sample_meas_noise).compile()
    flops_update = compiled_update.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Get SDE step counts
    sde_steps_per_meas = sol_stats['num_steps']
    total_sde_steps = int(jnp.sum(sde_steps_per_meas))
    mean_sde_steps = float(jnp.mean(sde_steps_per_meas))
    min_sde_steps = int(jnp.min(sde_steps_per_meas))
    max_sde_steps = int(jnp.max(sde_steps_per_meas))

    # Compute corrected FLOPS with decomposition
    flops_sde_total = flops_per_sde_step * total_sde_steps
    flops_update_total = flops_update * n_meas
    corrected_flops = flops_sde_total + flops_update_total

    return {
        "flops_per_ode_step": flops_per_sde_step,
        "flops_bayesian_per_meas": flops_update,
        "flops_ode_total": flops_sde_total,
        "flops_bayesian_total": flops_update_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_sde_steps,
        "mean_ode_steps_per_meas": mean_sde_steps,
        "min_ode_steps_per_meas": min_sde_steps,
        "max_ode_steps_per_meas": max_sde_steps,
        "ode_to_bayesian_ratio": flops_sde_total / flops_update_total if flops_update_total > 0 else float('inf'),
    }


def compute_decomposed_flops_particle(
    drift_fn: Callable,
    sample_samples: Array,
    update_fn: Callable,
    sample_meas: Array,
    n_meas: int,
    sol_stats: dict
) -> dict:
    """
    Compute FLOPS with decomposition for particle filter.

    Parameters
    ----------
    drift_fn : Callable
        JIT-compiled drift function for SDE.
    sample_samples : Array
        Sample particles array with correct shape.
    update_fn : Callable
        JIT-compiled weight update function.
    sample_meas : Array
        Single measurement array.
    n_meas : int
        Number of measurements.
    sol_stats : dict
        Statistics from diffrax solution containing 'num_steps'.

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information.
    """
    # Compile drift and get FLOPS for one SDE step
    compiled_drift = drift_fn.lower(0.0, sample_samples, None).compile()
    flops_per_sde_step = compiled_drift.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile weight update and get FLOPS
    compiled_update = update_fn.lower(sample_samples, sample_meas).compile()
    flops_update = compiled_update.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Get SDE step counts
    sde_steps_per_meas = sol_stats['num_steps']
    total_sde_steps = int(jnp.sum(sde_steps_per_meas))
    mean_sde_steps = float(jnp.mean(sde_steps_per_meas))
    min_sde_steps = int(jnp.min(sde_steps_per_meas))
    max_sde_steps = int(jnp.max(sde_steps_per_meas))

    # Compute corrected FLOPS with decomposition
    flops_sde_total = flops_per_sde_step * total_sde_steps
    flops_update_total = flops_update * n_meas
    corrected_flops = flops_sde_total + flops_update_total

    return {
        "flops_per_ode_step": flops_per_sde_step,
        "flops_bayesian_per_meas": flops_update,
        "flops_ode_total": flops_sde_total,
        "flops_bayesian_total": flops_update_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_sde_steps,
        "mean_ode_steps_per_meas": mean_sde_steps,
        "min_ode_steps_per_meas": min_sde_steps,
        "max_ode_steps_per_meas": max_sde_steps,
        "ode_to_bayesian_ratio": flops_sde_total / flops_update_total if flops_update_total > 0 else float('inf'),
    }


def compute_decomposed_flops_pgm(
    drift_fn: Callable,
    sample_samples: Array,
    update_fn: Callable,
    sample_assignments: Array,
    sample_meas: Array,
    n_meas: int,
    sol_stats: dict
) -> dict:
    """
    Compute FLOPS with decomposition for PGM filter.

    Parameters
    ----------
    drift_fn : Callable
        JIT-compiled drift function for SDE.
    sample_samples : Array
        Sample particles array with correct shape (n_samples, state_dim).
    update_fn : Callable
        JIT-compiled Kalman update function.
    sample_assignments : Array
        Sample cluster assignments (n_samples,).
    sample_meas : Array
        Single measurement array.
    n_meas : int
        Number of measurements.
    sol_stats : dict
        Statistics from diffrax solution containing 'num_steps'.

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information.
    """
    # Compile drift and get FLOPS for one SDE step
    compiled_drift = drift_fn.lower(0.0, sample_samples, None).compile()
    flops_per_sde_step = compiled_drift.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile Kalman update and get FLOPS
    compiled_update = update_fn.lower(sample_samples, sample_meas, jrandom.PRNGKey(0)).compile()
    flops_update = compiled_update.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Get SDE step counts
    sde_steps_per_meas = sol_stats['num_steps']
    total_sde_steps = int(jnp.sum(sde_steps_per_meas))
    mean_sde_steps = float(jnp.mean(sde_steps_per_meas))
    min_sde_steps = int(jnp.min(sde_steps_per_meas))
    max_sde_steps = int(jnp.max(sde_steps_per_meas))

    # Compute corrected FLOPS with decomposition
    flops_sde_total = flops_per_sde_step * total_sde_steps
    flops_update_total = flops_update * n_meas
    corrected_flops = flops_sde_total + flops_update_total

    return {
        "flops_per_ode_step": flops_per_sde_step,
        "flops_bayesian_per_meas": flops_update,
        "flops_ode_total": flops_sde_total,
        "flops_bayesian_total": flops_update_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_sde_steps,
        "mean_ode_steps_per_meas": mean_sde_steps,
        "min_ode_steps_per_meas": min_sde_steps,
        "max_ode_steps_per_meas": max_sde_steps,
        "ode_to_bayesian_ratio": flops_sde_total / flops_update_total if flops_update_total > 0 else float('inf'),
    }


def compute_decomposed_flops_pgm_em(
    drift_fn: Callable,
    sample_samples: Array,
    update_fn: Callable,
    sample_means_init: Array,
    sample_covs_init: Array,
    sample_weights_init: Array,
    sample_meas: Array,
    sample_prng_key: Array,
    n_meas: int,
    sol_stats: dict
) -> dict:
    """
    Compute FLOPS with decomposition for PGM-EM filter.

    Parameters
    ----------
    drift_fn : Callable
        JIT-compiled drift function for SDE.
    sample_samples : Array
        Sample particles array with correct shape (n_samples, state_dim).
    update_fn : Callable
        JIT-compiled update function (includes EM fitting + Kalman update).
    sample_means_init : Array
        Sample initial GMM means (cluster_dim, state_dim).
    sample_covs_init : Array
        Sample initial GMM covariances (cluster_dim, state_dim, state_dim).
    sample_weights_init : Array
        Sample initial GMM weights (cluster_dim,).
    sample_meas : Array
        Single measurement array.
    sample_prng_key : Array
        Sample PRNG key.
    n_meas : int
        Number of measurements.
    sol_stats : dict
        Statistics from diffrax solution containing 'num_steps'.

    Returns
    -------
    dict
        Dictionary containing decomposed FLOPS information.
    """
    # Compile drift and get FLOPS for one SDE step
    compiled_drift = drift_fn.lower(0.0, sample_samples, None).compile()
    flops_per_sde_step = compiled_drift.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Compile update (EM fitting + Kalman update) and get FLOPS
    compiled_update = update_fn.lower(sample_samples, sample_means_init, sample_covs_init, sample_weights_init, sample_meas, sample_prng_key).compile()
    flops_update = compiled_update.compiled.cost_analysis()[0].get("flops", float('nan'))

    # Get SDE step counts
    sde_steps_per_meas = sol_stats['num_steps']
    total_sde_steps = int(jnp.sum(sde_steps_per_meas))
    mean_sde_steps = float(jnp.mean(sde_steps_per_meas))
    min_sde_steps = int(jnp.min(sde_steps_per_meas))
    max_sde_steps = int(jnp.max(sde_steps_per_meas))

    # Compute corrected FLOPS with decomposition
    flops_sde_total = flops_per_sde_step * total_sde_steps
    flops_update_total = flops_update * n_meas
    corrected_flops = flops_sde_total + flops_update_total

    return {
        "flops_per_ode_step": flops_per_sde_step,
        "flops_bayesian_per_meas": flops_update,
        "flops_ode_total": flops_sde_total,
        "flops_bayesian_total": flops_update_total,
        "corrected_flops": corrected_flops,
        "n_meas": n_meas,
        "total_ode_steps": total_sde_steps,
        "mean_ode_steps_per_meas": mean_sde_steps,
        "min_ode_steps_per_meas": min_sde_steps,
        "max_ode_steps_per_meas": max_sde_steps,
        "ode_to_bayesian_ratio": flops_sde_total / flops_update_total if flops_update_total > 0 else float('inf'),
    }


def prepare_projection_filter(
    dynamic: DynamicalSystem,
    proj_filter_config: ProjectionFilterConfig,
    natural_statistics_symbolic: sp.Matrix,
    initial_samples: Array,
    theta_init: Array,
    params_init: tuple[Array, Array, Float],
    recompute_init_state: bool = True,
    simplify_lc: bool = True,
    ):
    """Prepare the projection filter by initializing parameters and propagating the flow.

    This function initializes the projection filter's natural parameters and bijection
    parameters, then propagates the convex conjugate flow to obtain updated parameters.

    Parameters
    ----------
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    proj_filter_config : ProjectionFilterConfig
        Configuration object containing the projection filter parameters.
    natural_statistics_symbolic : sp.Matrix
        Symbolic representation of the natural statistics.
    initial_samples : Array
        Initial samples for the projection filter.
    theta_init : Array
        Initial natural parameters for the projection filter.
    params_init : tuple[Array, Array, Float]
        Initial bijection parameters for the projection filter.

    Returns
    -------
    theta_init : Array
        Updated natural parameters after flow propagation.
    params_init : tuple[Array, Array, Float]
        Updated bijection parameters after flow propagation.
    psi_theta_init : Array
        Log partition function evaluated at the updated natural parameters.
    p : NDExponentialFamilySPG
        The exponential family object used in the projection filter.
    natural_statistics : Array
        Natural statistics converted to JAX.
    Lc_jax : Callable[[Array], Array]
        Backward Kolmogorov generator function in JAX.

    """

    x_sp = dynamic.sde.variables
    natural_statistics, _ = sympy_matrix_to_jax(natural_statistics_symbolic, x_sp)
    Lc_jax = nds.natural_statistics_backward_kolmogorov_gen_fun_jax_generator(natural_statistics_symbolic,
                                                                              dynamic.sde,
                                                                              {},
                                                                              simplify_lc)
    bijection = gauss_bijection_original
    p = NDExponentialFamilySPG(
        sample_space_dimension=dynamic.dim_states,
        sparse_grid_level=proj_filter_config.s_level,
        bijection=bijection,
        statistics=natural_statistics,
        s_rule=proj_filter_config.spg_rule,
        bijection_parameters=params_init,
        theta_indices_for_bijection_params=proj_filter_config.theta_indices_for_bijection_params,
        direct_calculation=proj_filter_config.direct_calculation,
    )

    eta_init = jnp.mean(p.natural_statistics(initial_samples.reshape(
        (initial_samples.shape[0] * initial_samples.shape[1],dynamic.dim_states))),
                        axis=0)
    theta_init = theta_init
    psi_theta_init = p.log_partition(theta_init, params_init)

    if recompute_init_state:
        try:
            sol = fl.propagate_convex_conjugate_cholesky_flow(p,
                                                            theta_init,
                                                            eta_init,
                                                            params_init,
                                                            proj_filter_config.theta_indices_for_bijection_params,
                                                            psi_theta_init,
                                                            proj_filter_config.learn_rate_init_prep,
                                                            proj_filter_config.alpha_prep,
                                                            proj_filter_config.beta_prep,
                                                            proj_filter_config.ode_solver_prep,
                                                            constant_step_size=False,
                                                            rtol=proj_filter_config.rtol,
                                                            atol=proj_filter_config.atol,
                                                            t1=proj_filter_config.t_f_prep,
                                                            dt_bayes=proj_filter_config.dt_prep,
                                                            max_steps=proj_filter_config.ode_max_steps)
        except Exception as e:
            logging.exception(e)
            raise

        theta_init = sol.ys[0][-1]
        params_init = (sol.ys[1][-1], sol.ys[2][-1], sol.ys[3][-1])
        psi_theta_init = p.log_partition(theta_init, params_init)




    return theta_init, params_init, psi_theta_init, p, natural_statistics, Lc_jax


def run_projection_filter(sim_config:SimulationConfig,
                          proj_filter_config: ProjectionFilterConfig,
                          p: NDExponentialFamily,
                          lc_jax: Callable[[Array, ], Array],
                          theta_init: Array,
                          params_init: tuple[Array, Array, Float],
                          measurements: Array,
                          progress_bar_type: str = "",
                          ):
    """
    Run the projection filter for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    proj_filter_config : ProjectionFilterConfig
        Configuration object containing the projection filter parameters.
    p : NDExponentialFamily
        The exponential family distribution used for filtering.
    lc_jax : Callable[[Array], Array]
        Backward Kolmogorov generator function in JAX.
    theta_init : Array
        Initial natural parameters.
    params_init : tuple[Array, Array, Float]
        Initial parameters for the distribution bijection.
    measurements : Array
        Sequence of measurements to filter.
    progress_bar_type : str, optional
        Whether to show progress bar during computation, defaults to no progress bar. The other options are text, and web.

    Returns
    -------
    tuple
        Results containing posterior parameters, learning rates and predicted parameters at each step.
    """

    # Define natural statistics and other necessary components

    # Time span for the simulation
    t_s = 0
    t_f = sim_config.t_f
    time_sample = (t_f - t_s) / sim_config.n_meas



    # Compile and run the cholesky projection filter
    compiled_cd_conjugate_proj_filter = cd_conj_proj_filter_chol.lower(p,
                                                                        lc_jax,
                                                                        theta_init,
                                                                        params_init,
                                                                        time_sample,
                                                                        measurements,
                                                                        get_theta_ell,
                                                                        proj_filter_config,
                                                                        progress_bar_type=progress_bar_type,
                                                                       ).compile()
    try:
        start_cdf = datetime.now()
        cdf_results = compiled_cd_conjugate_proj_filter(p,
                                                        lc_jax,
                                                                        theta_init,
                                                                        params_init,
                                                                        time_sample,
                                                                        measurements,
                                                                        get_theta_ell,
                                                                        proj_filter_config,
                                                                        progress_bar_type=progress_bar_type,
                                                        )
        cdf_results[0][0].block_until_ready()
        finish_cdf = datetime.now()
        execution_time_cdf = finish_cdf - start_cdf
    except Exception as e:
        logging.exception(e)
        # if there is an error in the execution, then exit
        raise

    # Create standalone functions for decomposed FLOPS analysis
    flow_derivative_fn = create_fokker_planck_flow_cholesky_for_cost_analysis(
            p, lc_jax,params_init[-1],proj_filter_config)

    bayesian_update_fn = create_bayesian_update_for_cost_analysis(
        p, get_theta_ell, proj_filter_config.theta_ell_args,
        proj_filter_config.mmt_iter
    )

    # Compute decomposed FLOPS
    fp_sol = cdf_results[5]
    sample_states = (theta_init, params_init[0], params_init[1])

    cd_proj_filter_cost_analysis = compute_decomposed_flops(
        flow_derivative_fn, sample_states,
        bayesian_update_fn, params_init, measurements[0],
        sim_config.n_meas, fp_sol.stats
    )

    return cdf_results, cd_proj_filter_cost_analysis, execution_time_cdf


def run_particle_filter(
        sim_config:SimulationConfig,
        dynamic: DynamicalSystem,
        measurements: Array,
        particle_filter_config: ParticleFilterConfig,
        init_samples: Array,
        prng_key: Array,
):
    """
    Run the particle filter for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    measurements : Array
        Array of measurements for the simulation.
    particle_filter_config : ParticleFilterConfig
        Configuration object containing the particle filter parameters.
    init_samples : Array
        Initial samples for the particles.
    prng_key : Array
        Pseudo-random number generator key for JAX.

    Returns
    -------
    x_particle_posterior_history : Array
        Resampled particle history (posterior) from the particle filter.
    x_particle_prior_history : Array
        Particle history (prior) from the particle filter.
    cd_particle_filter_cost_analysis : dict
        Cost analysis of the particle filter.
    execution_time_cd_par_filt : datetime.timedelta
        Execution time of the particle filter.
    """



    # Handle use_multi_core_cpu setting
    if sim_config.use_multi_core_cpu:
        n_devices = jax.local_device_count()
    else:
        n_devices = 1

    cd_par_filt = ContinuousDiscreteParticleFilter(
        n_devices=n_devices,
        n_particle_per_device=particle_filter_config.n_particle_per_device,
        initial_samples=init_samples,
        measurement_history=measurements,
        process_drift=dynamic.drift,
        process_diffusion=dynamic.diffusion,
        negative_likelihood=dynamic.negative_log_likelihood,
        process_brownian_dim=dynamic.dim_process_brownian,
        dt=sim_config.dt,
        dt_meas=(sim_config.t_f - 0) / sim_config.n_meas,
        prng_key=prng_key,
        sde_solver=particle_filter_config.sde_solver,
        use_stratonovich=particle_filter_config.use_stratonovich,
    )


    cd_par_filt_results = cd_par_filt.run()
    (_, _, x_particle_resampled_history, _, _, _, x_particle_history, _,
     execution_time_cd_par_filt, samples_sol_history) = cd_par_filt_results
    x_particle_posterior_history = x_particle_resampled_history.reshape(
        (-1, n_devices * particle_filter_config.n_particle_per_device, dynamic.dim_states))
    x_particle_prior_history = x_particle_history.reshape((-1, n_devices * particle_filter_config.n_particle_per_device,
                                                     dynamic.dim_states))

    # Create factory functions for decomposed FLOPS analysis
    drift_fn = create_particle_drift_for_cost_analysis(dynamic.drift)
    update_fn = create_particle_update_for_cost_analysis(dynamic.negative_log_likelihood)

    cd_particle_filter_cost_analysis = compute_decomposed_flops_particle(
        drift_fn=drift_fn,
        sample_samples=init_samples,
        update_fn=update_fn,
        sample_meas=measurements[0],
        n_meas=sim_config.n_meas,
        sol_stats=samples_sol_history.stats
    )
    return (x_particle_posterior_history, x_particle_prior_history, cd_particle_filter_cost_analysis,
            execution_time_cd_par_filt)


def run_enkf(sim_config:SimulationConfig,
             dynamic: DynamicalSystem,
             measurements: Array,
             enkf_config: EnKFConfig,
             init_samples: Array,
             prng_key: Array,):
    """
    Run the EnKF for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    measurements : Array
        Array of measurements for the simulation.
    enkf_config : EnKFConfig
        Configuration object containing the EnKF parameters.
    init_samples : Array
        Initial samples for the particles.
    prng_key : Array
        Pseudo-random number generator key for JAX.

    Returns
    -------
    enkf_results : tuple
        Results from the EnKF.
    enkf_cost_analysis : dict
        Cost analysis of the EnKF.
    execution_time_enkf : datetime.timedelta
        Execution time of the EnKF.
    """

    # Handle use_multi_core_cpu setting
    if sim_config.use_multi_core_cpu:
        n_devices = jax.local_device_count()
    else:
        n_devices = 1


    compiled_enkf = cd_enkf.lower(dynamic.drift,
                                  dynamic.diffusion,
                                  dynamic.measurement,
                                  dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                                  (sim_config.t_f - 0) / sim_config.n_meas,
                                  init_samples,
                                  measurements,
                                  n_devices,
                                  enkf_config.n_particle_per_device,
                                  dynamic.dim_process_brownian,
                                  prng_key,
                                  dt=sim_config.dt).compile()


    start_enkf = datetime.now()
    enkf_results = compiled_enkf(dynamic.drift,
                                  dynamic.diffusion,
                                  dynamic.measurement,
                                  dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                                  (sim_config.t_f - 0) / sim_config.n_meas,
                                  init_samples,
                                  measurements,
                                  n_devices,
                                  enkf_config.n_particle_per_device,
                                  dynamic.dim_process_brownian,
                                  prng_key,
                                  dt=sim_config.dt)

    enkf_results[0][0].block_until_ready()
    finish_enkf = datetime.now()
    execution_time_enkf = finish_enkf - start_enkf

    # enkf_results is (samples, samples_sol) where samples_sol is the diffeqsolve solution
    samples_history, samples_sol = enkf_results

    # Compute decomposed FLOPS
    drift_fn = create_enkf_drift_for_cost_analysis(dynamic.drift)
    update_fn = create_enkf_update_for_cost_analysis(
        dynamic.measurement,
        dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
        n_devices,
        enkf_config.n_particle_per_device
    )

    # Create sample arrays for cost analysis (measurement noise has dim_output dimensions)
    sample_meas_noise = jnp.zeros((n_devices, enkf_config.n_particle_per_device, dynamic.dim_output))

    enkf_cost_analysis = compute_decomposed_flops_ensemble(
        drift_fn, init_samples, update_fn,
        measurements[0], sample_meas_noise,
        sim_config.n_meas, samples_sol.stats
    )

    n_particle_per_device = init_samples.shape[1]
    enkf_particle_posterior_history = samples_history.reshape((-1, n_devices * n_particle_per_device, dynamic.dim_states))
    return enkf_particle_posterior_history, enkf_cost_analysis, execution_time_enkf


def run_gsf(
        sim_config:SimulationConfig,
        dynamic: DynamicalSystem,
        measurements: Array,
        means_gsf_init: Array,
        covs_gsf_init: Array,
        log_weights_gsf_init: Array,
    ):
    """
    Run the Gaussian Sum Filter (GSF) for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    measurements : Array
        Array of measurements for the simulation.
    means_gsf_init : Array
        Initial means for the Gaussian components in the GSF.
    covs_gsf_init : Array
        Initial covariances for the Gaussian components in the GSF.
    log_weights_gsf_init : Array
        Initial log weights for the Gaussian components in the GSF.

    Returns
    -------
    gsf_results : tuple
        Results from the GSF.
    gsf_cost_analysis : dict
        Cost analysis of the GSF.
    execution_time_gsf : datetime.timedelta
        Execution time of the GSF.
    """

    compiled_gsf = cd_gsf.lower(dynamic.drift,
                                dynamic.diffusion,
                                dynamic.measurement,
                                dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                                (sim_config.t_f - 0) / sim_config.n_meas,
                                means_gsf_init,
                                covs_gsf_init,
                                log_weights_gsf_init,
                                measurements).compile()


    start_gsf = datetime.now()
    gsf_results = compiled_gsf(dynamic.drift,
                                dynamic.diffusion,
                                dynamic.measurement,
                                dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                                (sim_config.t_f - 0) / sim_config.n_meas,
                                means_gsf_init,
                                covs_gsf_init,
                                log_weights_gsf_init,
                                measurements)

    gsf_results[0].block_until_ready()
    finish_gsf = datetime.now()
    execution_time_gsf = finish_gsf - start_gsf

    # gsf_results is (means, covs, log_weights, a_sol) where a_sol is the diffeqsolve solution
    a_sol = gsf_results[3]

    # Compute decomposed FLOPS
    ode_derivative_fn = create_ekf_ode_for_cost_analysis(dynamic.drift, dynamic.diffusion)
    update_fn = create_gsf_update_for_cost_analysis(
        dynamic.measurement,
        dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output)
    )

    gsf_cost_analysis = compute_decomposed_flops_gsf(
        ode_derivative_fn, (means_gsf_init, covs_gsf_init),
        update_fn, means_gsf_init, covs_gsf_init, log_weights_gsf_init,
        measurements[0], sim_config.n_meas, a_sol.stats
    )

    return gsf_results[:3], gsf_cost_analysis, execution_time_gsf


def run_sp_gsf(
        sim_config:SimulationConfig,
        dynamic: DynamicalSystem,
        measurements: Array,
        means_gsf_init: Array,
        covs_gsf_init: Array,
        log_weights_gsf_init: Array,
        sp_order:int,
    ):
    """
    Run the Sigma Point Gaussian Sum Filter (SP-GSF) for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    measurements : Array
        Array of measurements for the simulation.
    means_gsf_init : Array
        Initial means for the Gaussian components in the SP-GSF.
    covs_gsf_init : Array
        Initial covariances for the Gaussian components in the SP-GSF.
    log_weights_gsf_init : Array
        Initial log weights for the Gaussian components in the SP-GSF.
    sp_order : int
        Order of the sigma points.

    Returns
    -------
    sp_gsf_results : tuple
        Results from the SP-GSF.
    sp_gsf_cost_analysis : dict
        Cost analysis of the SP-GSF.
    execution_time_sp_gsf : datetime.timedelta
        Execution time of the SP-GSF.
    """

    sp = GaussHermiteSigmaPoints(n_state=dynamic.dim_states,
                                 order=sp_order)


    compiled_sp_gsf = cd_sp_gsf.lower(dynamic.drift,
                                      dynamic.diffusion,
                                      dynamic.measurement,
                                      dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                                      (sim_config.t_f - 0) / sim_config.n_meas,
                                      means_gsf_init,
                                      covs_gsf_init,
                                      log_weights_gsf_init,
                                      measurements,
                                      sp).compile()


    start_sp_gsf = datetime.now()
    sp_gsf_results = compiled_sp_gsf(dynamic.drift,
                                      dynamic.diffusion,
                                      dynamic.measurement,
                                      dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                                      (sim_config.t_f - 0) / sim_config.n_meas,
                                      means_gsf_init,
                                      covs_gsf_init,
                                      log_weights_gsf_init,
                                      measurements,
                                      sp)

    sp_gsf_results[0].block_until_ready()
    finish_sp_gsf = datetime.now()
    execution_time_sp_gsf = finish_sp_gsf - start_sp_gsf

    # sp_gsf_results is (means, covs, log_weights, a_sol) where a_sol is the diffeqsolve solution
    a_sol = sp_gsf_results[3]

    # Compute decomposed FLOPS
    ode_derivative_fn = create_spkf_ode_for_cost_analysis(dynamic.drift, dynamic.diffusion, sp)
    update_fn = create_spgsf_update_for_cost_analysis(
        dynamic.measurement,
        dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
        sp
    )

    sp_gsf_cost_analysis = compute_decomposed_flops_gsf(
        ode_derivative_fn, (means_gsf_init, covs_gsf_init),
        update_fn, means_gsf_init, covs_gsf_init, log_weights_gsf_init,
        measurements[0], sim_config.n_meas, a_sol.stats
    )

    return sp_gsf_results[:3], sp_gsf_cost_analysis, execution_time_sp_gsf


def run_pgm(
        sim_config:SimulationConfig,
        dynamic: DynamicalSystem,
        measurements: Array,
        means_gsf_init: Array,
        covs_gsf_init: Array,
        weights_pgm_init: Array,
        n_samples:int,
        prng_key: Array,
    ):
    """
    Run the PGM (K-Mean) for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    measurements : Array
        Array of measurements for the simulation.
    means_gsf_init : Array
        Initial means for the Gaussian components in the PGM.
    covs_gsf_init : Array
        Initial covariances for the Gaussian components in the PGM.
    weights_pgm_init : Array
        Initial weights for the Gaussian components in the PGM.
    n_samples : int
        Number of samples for the PGM.
    prng_key : Array
        Pseudo-random number generator key for JAX.

    Returns
    -------
    pgm_results : tuple
        Results from the PGM (K-Mean): (means, covs, weights).
    pgm_cost_analysis : dict
        Cost analysis of the PGM with corrected FLOPS.
    execution_time_pgm : datetime.timedelta
        Execution time of the PGM (K-Mean).
    """

    compiled_pgm = cd_pgm.lower(dynamic.drift,
                         dynamic.diffusion,
                         dynamic.measurement,
                         dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                         (sim_config.t_f - 0) / sim_config.n_meas,
                         means_gsf_init,
                         covs_gsf_init,
                         n_samples,
                         weights_pgm_init,
                         measurements,
                         dynamic.dim_process_brownian,
                         prng_key,
                         dt=sim_config.dt).compile()

    start_pgm = datetime.now()
    pgm_results = compiled_pgm(dynamic.drift,
                         dynamic.diffusion,
                         dynamic.measurement,
                         dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                         (sim_config.t_f - 0) / sim_config.n_meas,
                         means_gsf_init,
                         covs_gsf_init,
                         n_samples,
                         weights_pgm_init,
                         measurements,
                         dynamic.dim_process_brownian,
                         prng_key,
                         dt=sim_config.dt)

    pgm_results[0].block_until_ready()
    finish_pgm = datetime.now()
    execution_time_pgm = finish_pgm - start_pgm

    # pgm_results is (means, covs, weights, samples_sol) where samples_sol is the diffeqsolve solution
    samples_sol = pgm_results[3]

    # Create factory functions for decomposed FLOPS analysis
    cluster_dim = means_gsf_init.shape[0]
    state_dim = means_gsf_init.shape[1]
    drift_fn = create_pgm_drift_for_cost_analysis(dynamic.drift)
    update_fn = create_pgm_update_for_cost_analysis(
        dynamic.measurement,
        dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
        cluster_dim,
        state_dim,
        n_samples
    )

    # Create sample inputs for cost analysis
    sample_samples = jnp.zeros((n_samples, state_dim))
    sample_assignments = jnp.zeros(n_samples, dtype=jnp.int32)

    pgm_cost_analysis = compute_decomposed_flops_pgm(
        drift_fn=drift_fn,
        sample_samples=sample_samples,
        update_fn=update_fn,
        sample_assignments=sample_assignments,
        sample_meas=measurements[0],
        n_meas=sim_config.n_meas,
        sol_stats=samples_sol.stats
    )

    return pgm_results[:3], pgm_cost_analysis, execution_time_pgm

def run_pgm_em(sim_config:SimulationConfig,
        dynamic: DynamicalSystem,
        measurements: Array,
        means_gsf_init: Array,
        covs_gsf_init: Array,
        weights_pgm_init: Array,
        n_samples:int,
        prng_key: Array,
        em_max_iterations:int):
    """
    Run the PGM (EM) for the simulation.

    Parameters
    ----------
    sim_config : SimulationConfig
        Configuration object containing the simulation parameters.
    dynamic : DynamicalSystem
        The dynamical system to be simulated.
    measurements : Array
        Array of measurements for the simulation.
    means_gsf_init : Array
        Initial means for the Gaussian components in the PGM.
    covs_gsf_init : Array
        Initial covariances for the Gaussian components in the PGM.
    weights_pgm_init : Array
        Initial weights for the Gaussian components in the PGM.
    n_samples : int
        Number of samples for the PGM.
    prng_key : Array
        Pseudo-random number generator key for JAX.
    em_max_iterations : int
        Maximum number of EM iterations for the PGM.

    Returns
    -------
    pgm_em_results : tuple
        Results from the PGM (EM): (means, covs, weights).
    pgm_em_cost_analysis : dict
        Cost analysis of the PGM (EM) with corrected FLOPS.
    execution_time_pgm_em : datetime.timedelta
        Execution time of the PGM (EM).
    """
    try:
        compiled_pgm_em = cd_pgm_em.lower(dynamic.drift,
                         dynamic.diffusion,
                         dynamic.measurement,
                         dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                         (sim_config.t_f - 0) / sim_config.n_meas,
                         means_gsf_init,
                         covs_gsf_init,
                         n_samples,
                         weights_pgm_init,
                         measurements,
                         dynamic.dim_process_brownian,
                         prng_key,
                         em_max_iterations=em_max_iterations,
                         dt=sim_config.dt).compile()

        start_pgm_em = datetime.now()
        pgm_em_results = compiled_pgm_em(dynamic.drift,
                         dynamic.diffusion,
                         dynamic.measurement,
                         dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
                         (sim_config.t_f - 0) / sim_config.n_meas,
                         means_gsf_init,
                         covs_gsf_init,
                         n_samples,
                         weights_pgm_init,
                         measurements,
                         dynamic.dim_process_brownian,
                         prng_key,
                         em_max_iterations=em_max_iterations,
                         dt=sim_config.dt)

        pgm_em_results[0].block_until_ready()
        finish_pgm_em = datetime.now()
        execution_time_pgm_em = finish_pgm_em - start_pgm_em

        # pgm_em_results is (means, covs, weights, samples_sol) where samples_sol is the diffeqsolve solution
        samples_sol = pgm_em_results[3]

        # Create factory functions for decomposed FLOPS analysis
        cluster_dim = means_gsf_init.shape[0]
        state_dim = means_gsf_init.shape[1]
        drift_fn = create_pgm_em_drift_for_cost_analysis(dynamic.drift)
        update_fn = create_pgm_em_update_for_cost_analysis(
            dynamic.measurement,
            dynamic.parameters['sigma_v'] ** 2 * jnp.eye(dynamic.dim_output),
            cluster_dim,
            n_samples,
            em_max_iterations=em_max_iterations,
        )

        # Create sample inputs for cost analysis
        sample_samples = jnp.zeros((n_samples, state_dim))
        sample_means_init = jnp.zeros((cluster_dim, state_dim))
        sample_covs_init = jnp.tile(jnp.eye(state_dim), (cluster_dim, 1, 1))
        sample_weights_init = jnp.ones(cluster_dim) / cluster_dim
        sample_prng_key = jrandom.PRNGKey(0)

        pgm_em_cost_analysis = compute_decomposed_flops_pgm_em(
            drift_fn=drift_fn,
            sample_samples=sample_samples,
            update_fn=update_fn,
            sample_means_init=sample_means_init,
            sample_covs_init=sample_covs_init,
            sample_weights_init=sample_weights_init,
            sample_meas=measurements[0],
            sample_prng_key=sample_prng_key,
            n_meas=sim_config.n_meas,
            sol_stats=samples_sol.stats
        )

        return pgm_em_results[:3], pgm_em_cost_analysis, execution_time_pgm_em
    except Exception as e:
        logging.exception(e)
        pgm_em_results = []
        pgm_em_cost_analysis = {"corrected_flops": float('nan'), "raw_flops": float('nan')}
        execution_time_pgm_em = timedelta(0)

        return pgm_em_results, pgm_em_cost_analysis, execution_time_pgm_em
