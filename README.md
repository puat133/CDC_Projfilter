# Continuous-Discrete Projection Filter

[![arXiv](https://img.shields.io/badge/arXiv-2504.17324-b31b1b.svg)](https://arxiv.org/abs/2504.17324)

This repository is the companion code for the paper:

> **Conjugate Continuous-Discrete Projection Filter via Sparse-Grid Quadrature**
>
> Muhammad F. Emzir, Zaid A. Sawlan, Sami El Ferik
>
> [arXiv:2504.17324](https://arxiv.org/abs/2504.17324)

Implementation of the continuous-discrete projection filter for exponential family distributions with conjugate likelihoods. The method uses sparse-grid quadratures for numerical integration and information geometry techniques for projection onto the exponential family manifold. ODEs are solved using [diffrax](https://docs.kidger.site/diffrax/).

## Key Contributions

- **Local projection error analysis** for the prediction phase of the continuous-discrete projection filter
- **Exact Bayesian update** algorithm for discrete measurements with additive Gaussian noise
- **Superior performance** compared to state-of-the-art parametric continuous-discrete filtering methods (Particle Filter, EnKF, GSF, PGM)

## Features

- **Projection Filter** for exponential family distributions using sparse-grid quadratures
- **Benchmark Filters** for comparison:
  - Particle Filter (with systematic/multinomial resampling)
  - Ensemble Kalman Filter (EnKF)
  - Gaussian Sum Filter (GSF)
  - Projection Gaussian Mixture (PGM)
- **Cholesky parameterization** variant for improved numerical stability
- Support for multi-core CPU execution via JAX

## Installation

Requires Python 3.10+.

```bash
# Install core dependencies
pip install .

# Install with development tools
pip install ".[dev]"

# Install with notebook support
pip install ".[notebook]"

# Install everything
pip install ".[dev,notebook,viz]"
```

Or using uv:

```bash
uv pip install .
```

## Usage

### Van der Pol Oscillator Example

Run the main example with default parameters:

```bash
cd examples
python cdf_vdp.py
```

#### Command-line Options

```bash
python cdf_vdp.py \
    --seed=1 \
    --tf=1.0 \
    --dt=0.05 \
    --n_meas=4 \
    --sigmaw=1.0 \
    --sigmav=1.0 \
    --sim_case=1
```

#### Simulation Cases

| Case | Flag | Description |
|------|------|-------------|
| Projection Filter Only | `--sim_case=1` | Run only the projection filter |
| Benchmark Only | `--sim_case=2` | Run benchmark filters (PF, EnKF, GSF, PGM) |
| Full Comparison | `--sim_case=3` | Run all filters including projection filter |

#### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 1 | Random seed |
| `--tf` | 1.0 | Final simulation time |
| `--dt` | 0.05 | Sampling time step |
| `--n_meas` | 4 | Number of measurements |
| `--sigmaw` | 1.0 | Process noise scale |
| `--sigmav` | 1.0 | Measurement noise scale |
| `--s_level` | 8 | Sparse grid level |
| `--max_order` | 4 | Maximum monomial order |
| `--n_particle` | 1000000 | Number of particles (for PF/EnKF) |
| `--rtol` | 1e-3 | ODE solver relative tolerance |
| `--cholesky` | False | Use Cholesky parameterization |

### Running Multiple Seeds

```bash
cd examples
./cd_vdp_cholesky.sh 10 0  # Run 10 simulations starting from seed 1
```

## Project Structure

```text
├── cd_filtering/          # Core projection filter implementation
│   ├── cd_proj_conjugate.py   # Conjugate projection filter
│   ├── flows.py               # Fokker-Planck flow equations
│   └── bayesian_update.py     # Bayesian update step
├── simulation/            # Simulation framework
│   ├── simulation.py          # Main simulation orchestration
│   ├── runs.py                # Filter execution functions
│   ├── configs.py             # Configuration dataclasses
│   └── dynamical_system.py    # Dynamical system definition
├── exponential_family/    # Exponential family distributions
│   ├── n_d_ef.py              # N-dimensional exponential family
│   └── n_d_ef_spg.py          # Sparse-grid quadrature variant
├── sparse_quadrature/     # Sparse grid quadrature rules
│   ├── smolyak.py             # Smolyak sparse grids
│   ├── patterson.py           # Gauss-Patterson nodes
│   └── kronrod.py             # Gauss-Kronrod nodes
├── other_filter/          # Benchmark filter implementations
│   ├── particlefilter_cont_discrete.py
│   ├── cd_enkf.py
│   ├── cd_sp_kf.py            # Sigma-point Kalman filters
│   └── cd_pgm.py              # Projection Gaussian Mixture
├── symbolic/              # Symbolic computation utilities
│   ├── n_d.py                 # SDE and monomial definitions
│   └── sympy_to_jax.py        # SymPy to JAX conversion
├── sigma_points/          # Sigma point methods
├── ode_solver/            # Custom ODE solvers
├── utils/                 # Utility functions
├── examples/              # Example scripts
│   └── cdf_vdp.py             # Van der Pol oscillator example
└── test/                  # Test files
```

## Configuration Classes

### SimulationConfig

```python
from simulation.configs import SimulationConfig, Simulation_Case

sim_config = SimulationConfig(
    dt=0.05,
    t_f=1.0,
    n_meas=4,
    seed=1,
    sim_case=Simulation_Case.PROJECTION_FILTER_ONLY,
)
```

### ProjectionFilterConfig

```python
from simulation.configs import ProjectionFilterConfig

proj_config = ProjectionFilterConfig(
    theta_ell_args=(...),
    theta_init=theta_init,
    params_init=(means, covs, weights),
    theta_indices_for_bijection_params=(...),
    s_level=8,
    max_order_monomials=4,
    rtol=1e-3,
    cholesky_variant=False,
)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{emzir2025conjugate,
  title={Conjugate Continuous-Discrete Projection Filter via Sparse-Grid Quadrature},
  author={Emzir, Muhammad F. and Sawlan, Zaid A. and El Ferik, Sami},
  journal={arXiv preprint arXiv:2504.17324},
  year={2025}
}
```

## Historical Note

The following features have been removed from this codebase for maintainability:

- **SOS (Sum-of-Squares) Projection Filter**
- **Constrained Projection Filter**

For historical reference, see git tag `before-sos-removal`.

## License

see [LICENSE](LICENSE) for details.

## Authors

- Muhammad F. Emzir (<puat133@gmail.com>)
- Zaid A. Sawlan
- Sami El Ferik
