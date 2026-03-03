import jax
import jax.numpy as jnp
from jax import lax
import jax.scipy.linalg
from equinox import filter_jit
from jaxtyping import Array

@filter_jit
def regularized_natural_gradient(fisher: Array,
                                 v: Array,
                                 initial_lambda: float =1e-8,
                                 lambda_factor: float=4.0,
                                 max_attempts: int=32):
    """

    Parameters
    ----------
    fisher
    v
    initial_lambda
    lambda_factor
    max_attempts

    Returns
    -------

    """
    # Symmetrize matrix
    g_sym = (fisher + fisher.T) / 2.0
    eye = jnp.eye(fisher.shape[0])

    def solve_system(lam):
        """Solve system with regularization and return (solution, success)"""
        regularized_fisher = g_sym + lam * eye
        chol = jax.scipy.linalg.cholesky(regularized_fisher, lower=True)
        valid = jnp.all(chol.diagonal() > 0)  # Check PD via Cholesky factors

        # Only compute solution if valid
        solution = jax.lax.cond(
            valid,
            lambda: jax.scipy.linalg.cho_solve((chol, True), v),
            lambda: jnp.full_like(v, jnp.nan)
        )
        return solution, valid

    # First attempt without regularization
    initial_sol, initial_success = solve_system(0.0)

    # Only run scan if initial attempt failed
    def run_scan(_):
        lambdas = initial_lambda * lambda_factor ** jnp.arange(max_attempts)

        def scan_fn(carry, lam):
            current_sol, _ = carry
            new_sol, success = solve_system(lam)
            return (jnp.where(success, new_sol, current_sol), success), None

        (final_sol, _), _ = lax.scan(
            scan_fn,
            (jnp.full_like(v, jnp.nan), False),
            lambdas
        )
        return final_sol

    # Choose between initial solution and scan result
    return lax.cond(
        initial_success,
        lambda _: initial_sol,
        run_scan,
        operand=None
    )

@filter_jit
def eigenvalue_truncation_natural_gradient(fisher: Array,
                                          v: Array,
                                          min_eigenvalue: float = 0.0,
                                          max_norm: float = jnp.inf) -> tuple[Array, bool]:
    """
    Solve Fisher^{-1} @ v using eigenvalue decomposition with truncation.

    This method computes the eigendecomposition F = V @ diag(λ) @ V^T and sets
    small eigenvalues below a threshold to a minimum value to avoid division by zero.
    This is a hard truncation approach (spectral filtering).

    Parameters
    ----------
    fisher : Array
        Fisher information matrix of shape (n, n)
    v : Array
        Right-hand side vector of shape (n,)
    min_eigenvalue : float, optional
        Minimum eigenvalue threshold. Eigenvalues below this are set to this value.
        Default is 0.0.

    Returns
    -------
    Tuple[Array, bool]
        A tuple containing:
        - solution (Array): solution vector x that approximates F^{-1} @ v
        - success (bool): whether computation succeeded (always True unless NaN/Inf)

    Notes
    -----
    **USE WITH CAUTION for auto-diff Fisher matrices**: This method uses hard truncation
    with a sharp cutoff, which can lead to discontinuities in the solution. For Fisher
    matrices computed via automatic differentiation, prefer regularized_natural_gradient
    (TIKHONOV_CHOLESKY) for more robust handling of near-singularities.

    This method is more suitable for sampling-based Fisher approximations where eigenvalues
    can be structurally small or negative.

    References
    ----------
    - Regularization by spectral filtering (Wikipedia)
    - MIT 9.520 Lecture Notes on Spectral Regularization
    """
    # Symmetrize fisher matrix
    g_sym = (fisher + fisher.T) / 2.0

    # Compute eigendecomposition
    fisher_ev, fisher_V = jnp.linalg.eigh(g_sym)

    # Truncate small eigenvalues
    # fisher_ev_safe = jnp.maximum(fisher_ev, min_eigenvalue)
    # fisher_ev_reciprocal = 1.0 / fisher_ev_safe
    # fisher_ev_reciprocal is either 1/fisher_ev if fisher_ev> min_eigenvalue, or 0 otherwise
    fisher_ev_reciprocal = jnp.where(fisher_ev>min_eigenvalue, 1/fisher_ev, 0)

    # Compute solution: V @ diag(1/λ) @ V^T @ v
    solution = fisher_V @ (fisher_ev_reciprocal * (fisher_V.T @ v))

    solution_norm = jnp.linalg.norm(solution)
    # scaling to ensure that the solution norm is less than equal to the max_norm
    solution = solution * jnp.minimum(solution_norm,max_norm)/solution_norm


    return solution


@filter_jit
def tikhonov_regularized_natural_gradient(fisher: Array,
                                          v: Array,
                                          min_eigenvalue: float = 0.0,
                                          max_norm: float = jnp.inf) -> tuple[Array, bool]:
    
    # Symmetrize fisher matrix
    g_sym = (fisher + fisher.T) / 2.0
    solution = jnp.linalg.solve(g_sym + min_eigenvalue * jnp.eye(g_sym.shape[0]),v)
    solution_norm = jnp.linalg.norm(solution)
    # scaling to ensure that the solution norm is less than equal to the max_norm
    solution = solution * jnp.minimum(solution_norm,max_norm)/solution_norm


    return solution