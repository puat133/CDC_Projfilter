import jax.numpy as jnp
from functools import partial
from jaxtyping import Array

@partial(jnp.vectorize, signature='(n,n)->(),()')
def min_max_eigvalsh(a_hermittian_matrix: Array):
    eigvals = jnp.linalg.eigvalsh(a_hermittian_matrix)
    return jnp.min(eigvals), jnp.max(eigvals)


@partial(jnp.vectorize, signature='(n),(n)->()')
def inner(a: Array, b: Array) -> float:
    return jnp.inner(a, b)

@partial(jnp.vectorize, signature='(n),(n)->(n,n)')
def outer(a: Array, b: Array) -> Array:
    return jnp.outer(a, b)


@partial(jnp.vectorize, signature='(m,n),(n)->(m)')
def mat_vec(a: Array, b: Array):
    """
    matrix vector product
    Parameters
    ----------
    a : Array
        (m, n)
    b : Array
        (n,)

    Returns
    -------
    result: Array
        (m,)
    """
    return a @ b

def vecsym(G: Array):
    """
    Given a symmetric MxM matrix G, return the vector of its upper-triangular
    entries (including the diagonal), in row-major order.

    Parameters
    ----------
    G : Array (mxm)
        Symmetric matrix to be flattened.

    Returns
    -------
    g : Array (length m(m+1)//2)
        Flattened vector of the upper-triangular entries of G in row-major order.
    """
    # triu_indices_from(G) returns two arrays (row_idx, col_idx) that index
    # the upper-triangle of G, in row-major order
    row_idx, col_idx = jnp.triu_indices_from(G)

    # Take those entries from G
    g = G[row_idx, col_idx]
    return g

def vecsym_inv(vec: Array, m: int):
    """
    Given a 1D array 'vec' representing the upper-triangular entries
    (including the diagonal) of an MxM symmetric matrix in row-major order,
    reconstruct and return the full MxM symmetric matrix.

    Parameters
    ----------
    vec : Array, shape (M*(M+1)//2,)
        The flattened upper-triangular entries of some symmetric MxM matrix.
    m : int
        The size of the original symmetric matrix.

    Returns
    -------
    G : Array, shape (M, M)
        The reconstructed symmetric matrix.
    """
    # 1) Create an empty MxM array
    G = jnp.zeros((m, m), dtype=vec.dtype)

    # 2) Get the indices for the upper triangle (including diagonal)
    r, c = jnp.triu_indices(m)

    # 3) Place 'vec' entries into the upper triangle of G
    G = G.at[r, c].set(vec)

    # 4) "Reflect" the upper triangle to fill the lower triangle
    #    A convenient approach is to add G's transpose and then
    #    subtract the diagonal (since it would be doubled otherwise).
    G = G + G.T - jnp.diag(jnp.diag(G))

    return G