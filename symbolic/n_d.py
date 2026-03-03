import sympy as sp
import numpy as onp
import jax.numpy as jnp
from itertools import chain

from jax import numpy as jnp
from jaxtyping import Array
from symbolic.one_d import SDE
from symbolic.sympy_to_jax import sympy_matrix_to_jax
from utils.vectorized import vecsym, vecsym_inv


def mix_monomials_up_to_order(x: tuple, max_order: int, norm_order: int = 1) -> sp.MutableDenseMatrix:
    """
    Generate a matrix containing monomials with order up to max_order

    Parameters
    ----------
    x : tuple
        Tuple of symbols x1,...,xd representing variables
    max_order : int
        Maximum order of monomials to generate
    norm_order : int, optional
        Norm order used to decide which monomials to include, by default 1

    Returns
    -------
    sp.Matrix
        Matrix containing monomials excluding constant 1, where each monomial
        is included if its norm order is less than max_order + 1

    Notes
    -----
    Uses the specified norm_order to select monomials whose order is less than
    max_order + 1 when computing their norm. The resulting monomials are sorted
    and returned as a sympy Matrix.
    """
    dim = len(x)
    order = onp.arange(max_order + 1)
    orders = onp.stack(onp.meshgrid(*onp.tile(order, (dim, 1))), axis=-1)
    orders_lined = orders.reshape(-1, dim)
    selected_order = orders_lined[onp.linalg.norm(orders_lined, ord=norm_order,
                                                  axis=-1) < max_order + 1]
    selected_order_list = list(map(tuple, selected_order))  # convert to list of tuple
    selected_order_list.sort()
    monomials = []
    x_ = onp.array(x)
    for i in range(1, selected_order.shape[0]):
        monomials.append(sp.prod(onp.power(x_, selected_order_list[i])))
    return sp.Matrix(monomials)


def hyperbolic_cross_monomials(x: tuple, max_order: int) -> sp.MutableDenseMatrix:
    """
    Generate a matrix containing monomials with order up to max_order using hyperbolic cross indexing.

    This function generates monomials using a hyperbolic cross selection criteria, which helps
    reduce the number of terms while maintaining accuracy for high-dimensional problems.

    Parameters
    ----------
    x : tuple
        Tuple of sympy symbols x1,...,xd representing variables.
    max_order : int
        Maximum order for hyperbolic cross monomial selection.

    Returns
    -------
    sp.Matrix
        Matrix of monomials selected using hyperbolic cross criteria, excluding constant 1.

    Notes
    -----
    Uses hyperbolic cross criteria to select monomials, where the product of max(1,orders)
    for each variable must be less than max_order + 1. This produces a sparser set of
    monomials compared to standard polynomial bases while maintaining good approximation
    properties.
    """
    dim = len(x)
    order = onp.arange(max_order + 1)
    orders = onp.stack(onp.meshgrid(*onp.tile(order, (dim, 1))), axis=-1)
    max_1_orders = onp.maximum(1, orders)
    max_1_orders_lined = max_1_orders.reshape(-1, dim)
    orders_lined = orders.reshape(-1, dim)
    selected_order = orders_lined[onp.prod(max_1_orders_lined, axis=-1) < max_order + 1]
    selected_order_list = list(map(tuple, selected_order))  # convert to list of tuple
    selected_order_list.sort()  # convert to list of tuple
    monomials = []
    x_ = onp.array(x)
    for i in range(1, selected_order.shape[0]):
        monomials.append(sp.prod(onp.power(x_, selected_order_list[i])))
    return sp.Matrix(monomials)


def backward_diffusion(fun_array: sp.MutableDenseMatrix, sde: SDE) \
        -> sp.MutableDenseMatrix:
    """
    Compute backward diffusion operator of a given function for a given sde. The function and sde
    should be function of the same symbolic variable.

    Parameters
    ----------
    fun_array : Callable[[sympy.Symbol], sympy.Symbol]
    sde : SDE

    Returns
    -------
    res : sympy.matrices.dense.MutableDenseMatrix
    """
    jac = fun_array.jacobian(sde.variables)
    res = jac * sde.drifts
    for i in range(jac.shape[0]):
        res[i] += sp.trace(sde.diffusions.transpose() * jac[i, :].jacobian(sde.variables) * sde.diffusions) / 2
    return res


def get_monomial_degree_set(fun_array: sp.MutableDenseMatrix, variables: tuple[sp.Symbol]) -> set[tuple[int]]:
    """
    Get the set of monomial degrees from an array of polynomials.

    This function extracts the degrees of each monomial term appearing in a matrix
    of polynomial expressions.

    Parameters
    ----------
    fun_array : sp.MutableDenseMatrix
        Matrix containing polynomial expressions to analyze
    variables : tuple[sp.Symbol]
        Tuple of sympy symbols representing the variables in the polynomials

    Returns
    -------
    set[tuple[int]]
        Set of tuples containing the degrees of each monomial term. Each tuple has
        length equal to number of variables, with entries representing the degree
        of each variable in that monomial.

    Notes
    -----
    For constant terms or zero polynomials, returns a tuple of zeros with length
    equal to number of variables.
    """
    monomial_degree_set = set()
    n_states = len(variables)
    zero_monom = tuple([0 for i in range(n_states)])

    for an_entry in fun_array:
        a_poly = an_entry.as_poly(variables)
        if a_poly:
            monoms = an_entry.as_poly(variables).monoms()
        else:
            monoms = [zero_monom]
        for monom in monoms:
            monomial_degree_set.add(monom)

    return monomial_degree_set


def from_tuple_to_symbolic_monom(variables: tuple[sp.Symbol], degree_tuple: tuple[int]) -> sp.Symbol:
    """
    Convert a tuple of variable degrees into a symbolic monomial.

    Parameters
    ----------
    variables : tuple[sp.Symbol]
        Tuple of symbolic variables (x1, x2, ..., xn)
    degree_tuple : tuple[int]
        Tuple specifying the degree of each variable in the monomial

    Returns
    -------
    sp.Symbol
        Symbolic monomial expression formed by raising each variable to its
        corresponding degree and multiplying the terms together

    Examples
    --------
    >>> variables = (x1, x2)
    >>> degree_tuple = (2, 1)
    >>> from_tuple_to_symbolic_monom(variables, degree_tuple)
    x1**2 * x2

    Notes
    -----
    Creates a monomial by iterating through the variables and degrees in parallel,
    raising each variable to its specified power and multiplying the terms together.
    A degree of 0 for a variable means that variable does not appear in the result.
    """
    res = 1
    for i in range(len(degree_tuple)):
        res *= variables[i] ** degree_tuple[i]
    return res


def column_polynomials_coefficients(col: sp.MutableDenseMatrix, variables: tuple[sp.Symbol],
                                    monomial_degree_list: list[tuple[int, int]] = None) -> \
        tuple[list[sp.Symbol], onp.ndarray]:
    """
    Calculate the coefficients matrix for a column vector of polynomials with respect to a given set of monomials.

    Parameters
    ----------
    col : sp.MutableDenseMatrix
        Column vector of polynomial expressions.
    variables : tuple[sp.Symbol]
        Tuple of symbolic variables used in the polynomials.
    monomial_degree_list : list[tuple[int, int]], optional
        List of monomial degrees to use as basis. If None, will extract degrees from col.

    Returns
    -------
    tuple[list[sp.Symbol], np.ndarray]
        Returns tuple containing:
            - List of symbolic monomial expressions.
            - Coefficient matrix with shape (col.shape[0], len(monoms)) where entry (i,j)
                is coefficient of monomial j in polynomial i.

    Notes
    -----
    Each polynomial in col is represented as a linear combination of monomials from monomial_degree_list.
    For each polynomial and monomial, finds the coefficient of that monomial in the polynomial and stores it
    in the coefficient matrix.
    """


    if not monomial_degree_list:
        monomial_degree_list = get_monomial_degree_set(col, variables)

    monoms = [from_tuple_to_symbolic_monom(variables, monomial_degree) for monomial_degree in monomial_degree_list]
    coefficients = onp.zeros((col.shape[0], len(monoms)))
    for i in range(col.shape[0]):
        for j in range(len(monoms)):
            a_poly = col[i].as_poly(variables)
            if a_poly:
                coefficients[i, j] = a_poly.coeff_monomial(monoms[j])

    return monoms, coefficients


def compute_f_2_and_f_4(x: sp.Matrix,
                        h: sp.Matrix,
                        R: sp.Matrix,
                        gamma: sp.Matrix):
    n = x.shape[0]
    cal_F2 = sp.Matrix(jnp.zeros((h.shape[0] * gamma.shape[1],), dtype=jnp.int32))
    vect_R = sp.Matrix(R.flat())
    cal_F4 = sp.kronecker_product(vect_R, h)
    for i in range(n):
        Gi = gamma[i, :].transpose()
        cal_F2 += sp.kronecker_product(sp.diff(h, x[i]), Gi)
        cal_F4 += sp.kronecker_product(sp.diff(vect_R, x[i]), Gi)

    return cal_F2, cal_F4


# since m = 1, then kronecker product equal to just ordinary multiplication
def compute_f_0_1_3(x: sp.Matrix,
                    h: sp.Matrix,
                    natural_statistic: sp.Matrix,
                    gamma: sp.Matrix):
    """
    Calculate the F_0, F_1, and F_3 matrices from Ref[1] Section 4.

    Parameters
    ----------
    x : sp.Matrix
        State variables.
    h : sp.Matrix
        Measurement drift vector.
    natural_statistic : sp.Matrix
        Matrix of natural statistics.
    gamma : sp.Matrix
        Cross-covariance matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - cal_F0 : sp.Matrix
            F0 matrix for projection filter.
        - cal_F1 : sp.Matrix
            F1 matrix for projection filter.
        - cal_F3 : sp.Matrix
            F3 matrix for projection filter.

    References
    ----------
    [1] M. F. Emzir, "Projection filter algorithm for correlated measurement and process noises,
    with state dependent covariances"
    """

    n = x.shape[0]
    cal_F0 = sp.Matrix(jnp.zeros((h.shape[0] * gamma.shape[1],), dtype=jnp.int32))
    cal_F1 = sp.Matrix(jnp.zeros((gamma.shape[1],), dtype=jnp.int32))
    cal_F3 = sp.Matrix(jnp.zeros((gamma.shape[1],), dtype=jnp.int32))

    for i in range(n):
        Gi = gamma[i, :].transpose()
        cal_F0 += (sp.kronecker_product(Gi, h) + sp.kronecker_product(h, Gi)) * sp.diff(natural_statistic, x[i])
        cal_F0 += sp.kronecker_product(sp.diff(h, x[i]), Gi) * natural_statistic
        cal_F1 += Gi * sp.diff(natural_statistic, x[i])
        cal_F3 += sp.kronecker_product(Gi, sp.diff(natural_statistic, x[i]))
        for j in range(n):
            Gj = gamma[j, :].transpose()
            cal_F0 += sp.kronecker_product(sp.diff(Gi, x[j]), Gj) * sp.diff(natural_statistic, x[i])
            cal_F0 += sp.kronecker_product(Gi, Gj) * sp.diff(sp.diff(natural_statistic, x[i]), x[j])

    return cal_F0, cal_F1, cal_F3


def compute_capital_f_of_statistics(x: sp.Matrix,
                                    h: sp.Matrix,
                                    R: sp.Matrix,
                                    natural_statistics: sp.Matrix,
                                    gamma: sp.Matrix):
    """
    Compute the capital F matrices for the given natural statistics.

    This function calculates the capital F matrices (F0, F1, F2, F3, F4) used in the projection filter
    algorithm for a given set of natural statistics, measurement drift vector, measurement noise covariance,
    and cross-covariance matrix.

    Parameters
    ----------
    x : sp.Matrix
        State variables.
    h : sp.Matrix
        Measurement drift vector.
    R : sp.Matrix
        Measurement noise covariance matrix.
    natural_statistics : sp.Matrix
        Matrix of natural statistics.
    gamma : sp.Matrix
        Cross-covariance matrix.

    Returns
    -------
    tuple
        A tuple containing:
        - cal_F0 : sp.Matrix
            F0 matrix for projection filter.
        - cal_F1 : sp.Matrix
            F1 matrix for projection filter.
        - cal_F2 : sp.Matrix
            F2 matrix for projection filter.
        - cal_F3 : sp.Matrix
            F3 matrix for projection filter.
        - cal_F4 : sp.Matrix
            F4 matrix for projection filter.

    Notes
    -----
    The function first computes the F2 and F4 matrices using the measurement drift vector, measurement noise
    covariance matrix, and cross-covariance matrix. It then calculates the F0, F1, and F3 matrices for each
    natural statistic using the state variables, measurement drift vector, natural statistics, and cross-covariance
    matrix. The resulting matrices are returned as a tuple.
    """

    cal_F2, cal_F4 = compute_f_2_and_f_4(x, h, R, gamma)
    cal_F0_list = []
    cal_F1_list = []
    cal_F3_list = []
    for i in range(len(natural_statistics)):
        c = natural_statistics[i, :]
        cal_F0_, cal_F1_, cal_F3_ = compute_f_0_1_3(x, h, c, gamma)
        cal_F0_list.append([cal_F0_.T])
        cal_F1_list.append([cal_F1_.T])
        cal_F3_list.append([cal_F3_.T])

    cal_F0 = sp.Matrix(cal_F0_list)
    cal_F1 = sp.Matrix(cal_F1_list)
    cal_F3 = sp.Matrix(cal_F3_list)

    return cal_F0, cal_F1, cal_F2, cal_F3, cal_F4


def get_ito_vector_projection_filter_statistics(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                                dynamic_sde: SDE,
                                                measurement_sde: SDE,
                                                simplified=False
                                                ):
    """
    Calculate the Ito vector projection filter statistics.

    This function computes the necessary statistics for the Ito vector projection filter
    using the provided natural statistics, dynamic SDE, and measurement SDE.

    Parameters
    ----------
    natural_statistics_symbolic : sp.MutableDenseMatrix
        Matrix of natural statistics in symbolic form.
    dynamic_sde : SDE
        SDE object containing drift and diffusion terms of the dynamics.
    measurement_sde : SDE
        SDE object containing drift and diffusion terms of the measurements.
    simplified : bool, optional
        If True, the returned statistics will be simplified using sympy.simplify, by default False

    Returns
    -------
    tuple
        A tuple containing:
        - h : sp.Matrix
            Measurement drift vector.
        - hh : sp.Matrix
            Outer product of measurement drift vector with itself.
        - hc : sp.Matrix
            Product of natural statistics and measurement drift vector.
        - hhc : sp.Matrix
            Product of natural statistics and the Kronecker product of measurement drift vector with itself.
        - Lc : sp.Matrix
            Backward diffusion operator applied to natural statistics.

    Notes
    -----
    The function first computes the backward diffusion operator for the natural statistics
    using the dynamic SDE. It then calculates various statistics involving the measurement
    drift terms. If `simplified` is True, the returned statistics will be simplified using
    sympy.simplify.
    """

    x = dynamic_sde.variables
    c = natural_statistics_symbolic
    Lc = backward_diffusion(natural_statistics_symbolic, dynamic_sde)
    h = measurement_sde.drifts
    hc = c * h.transpose()
    hh = h.transpose() * h
    hhc = (h.transpose() * h)[0] * c

    if not simplified:
        return h, hh, hc, hhc, Lc
    else:
        return sp.simplify(h), sp.simplify(hh), sp.simplify(hc), sp.simplify(hhc), sp.simplify(Lc)


def get_projection_filter_statistics_correlated(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                                dynamic_sde: SDE,
                                                measurement_sde: SDE,
                                                cross_covariance_matrix: Array,
                                                simplified=False
                                                ):
    """
    Calculate the projection filter statistics for correlated noise.

    This function computes the necessary statistics for the projection filter
    when the process and measurement noises are correlated. It uses the provided
    natural statistics, dynamic SDE, measurement SDE, and cross-covariance matrix.

    Parameters
    ----------
    natural_statistics_symbolic : sp.MutableDenseMatrix
        Matrix of natural statistics in symbolic form.
    dynamic_sde : SDE
        SDE object containing drift and diffusion terms of the dynamics.
    measurement_sde : SDE
        SDE object containing drift and diffusion terms of the measurements.
    cross_covariance_matrix : Array
        Cross-covariance matrix representing the correlation between process and measurement noise.
    simplified : bool, optional
        If True, the returned statistics will be simplified using sympy.simplify, by default False

    Returns
    -------
    tuple
        A tuple containing:
        - h : sp.Matrix
            Measurement drift vector.
        - hh : sp.Matrix
            Outer product of measurement drift vector with itself.
        - hc : sp.Matrix
            Product of natural statistics and measurement drift vector.
        - hhc : sp.Matrix
            Product of natural statistics and the Kronecker product of measurement drift vector with itself.
        - R : sp.Matrix
            Measurement noise covariance matrix.
        - Lc : sp.Matrix
            Backward diffusion operator applied to natural statistics.
        - cal_F0 : sp.Matrix
            F0 matrix for projection filter.
        - cal_F1 : sp.Matrix
            F1 matrix for projection filter.
        - cal_F2 : sp.Matrix
            F2 matrix for projection filter.
        - cal_F3 : sp.Matrix
            F3 matrix for projection filter.
        - cal_F4 : sp.Matrix
            F4 matrix for projection filter.

    Notes
    -----
    The function first computes the backward diffusion operator for the natural statistics
    using the dynamic SDE. It then calculates various statistics involving the measurement
    drift and diffusion terms, as well as the cross-covariance matrix. Finally, it computes
    the F matrices needed for the projection filter.
    """

    x = dynamic_sde.variables
    c = natural_statistics_symbolic
    Lc = backward_diffusion(natural_statistics_symbolic, dynamic_sde)
    h = measurement_sde.drifts
    hc = c * h.transpose()
    hh = h * h.transpose()
    hhc = c * (sp.kronecker_product(h, h)).transpose()
    R = measurement_sde.diffusions * measurement_sde.diffusions.transpose()
    S = sp.Matrix(cross_covariance_matrix.tolist())
    gamma = dynamic_sde.diffusions * S * measurement_sde.diffusions.transpose()
    cal_F0, cal_F1, cal_F2, cal_F3, cal_F4 = compute_capital_f_of_statistics(sp.Matrix(list(x)), h, R, c, gamma)

    if not simplified:
        return h, hh, hc, hhc, R, Lc, cal_F0, cal_F1, cal_F2, cal_F3, cal_F4
    else:
        return sp.simplify(h), sp.simplify(hh), sp.simplify(hc), sp.simplify(hhc), sp.simplify(R), sp.simplify(Lc), \
            sp.simplify(cal_F0), sp.simplify(cal_F1), sp.simplify(cal_F2), sp.simplify(cal_F3), sp.simplify(cal_F4)


def get_projection_filter_matrices_correlated(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                              dynamic_sde: SDE,
                                              measurement_sde: SDE,
                                              cross_covariance_matrix: Array,
                                              ) -> tuple[list[onp.ndarray], list[tuple], list[tuple]]:
    """
    Calculate the projection filter matrices for correlated noise.

    This function computes the projection filter matrices for a system with correlated noise
    using the provided natural statistics, dynamic SDE, measurement SDE, and cross-covariance matrix.

    Parameters
    ----------
    natural_statistics_symbolic : sp.MutableDenseMatrix
        Matrix of natural statistics in symbolic form.
    dynamic_sde : SDE
        SDE object containing drift and diffusion terms of the dynamics.
    measurement_sde : SDE
        SDE object containing drift and diffusion terms of the measurements.
    cross_covariance_matrix : Array
        Cross-covariance matrix representing the correlation between process and measurement noise.

    Returns
    -------
    tuple
        A tuple containing:
        - List of projection filter matrices [F0, F1, F2, F3, F4, AR, A, H1, H2]
        - List of monomial degrees for all monomials used in the projection filter
        - List of monomial degrees for the remaining monomials after removing natural statistics

    Notes
    -----
    The function first computes the necessary statistics and monomial sets, then calculates the
    coefficients for the projection filter matrices. It ensures that the constant monomial is
    handled appropriately and constructs the final list of monomials used in the projection filter.
    """
    x = dynamic_sde.variables
    c = natural_statistics_symbolic
    h, hh, hc, hhc, R, Lc, cal_F0, cal_F1, cal_F2, cal_F3, cal_F4 = \
        get_projection_filter_statistics_correlated(natural_statistics_symbolic,
                                                    dynamic_sde,
                                                    measurement_sde,
                                                    cross_covariance_matrix)

    natural_monom_set = get_monomial_degree_set(c, x)
    monom_set = natural_monom_set.union(get_monomial_degree_set(Lc, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F0, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F1, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F2, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F3, x))
    monom_set = monom_set.union(get_monomial_degree_set(cal_F4, x))
    remaining_monoms_set = monom_set.difference(natural_monom_set)

    constant_monom = tuple([0 for x_ in x])
    if constant_monom in remaining_monoms_set:
        remaining_monoms_set.remove(constant_monom)
        # constant is removed from remaining monoms_set temporarily but will be put back at its end

    natural_monom_list = list(natural_monom_set)
    natural_monom_list.sort()
    remaining_monoms_list = list(remaining_monoms_set)
    remaining_monoms_list.sort()
    remaining_monoms_list.append(constant_monom)  # we put back the constant monom here
    monom_list = list(chain.from_iterable(
        [natural_monom_list, remaining_monoms_list]))

    _, A = column_polynomials_coefficients(Lc.vec(), x, monom_list)

    F0 = []
    F1 = []
    for i in range(len(c)):
        _, F0_ = column_polynomials_coefficients(cal_F0[i, :].vec(), x, monom_list)
        _, F1_ = column_polynomials_coefficients(cal_F1[i, :].vec(), x, monom_list)

        F0.append(F0_)
        F1.append(F1_)

    F0 = jnp.stack(F0)
    F1 = jnp.stack(F1)

    _, F2 = column_polynomials_coefficients(cal_F2.vec(), x, monom_list)
    _, F3 = column_polynomials_coefficients(cal_F3.vec(), x, monom_list)
    _, F4 = column_polynomials_coefficients(cal_F4.vec(), x, monom_list)
    _, AR = column_polynomials_coefficients(R.vec(), x, monom_list)
    _, H1 = column_polynomials_coefficients(h.vec(), x, natural_monom_list)
    _, H2 = column_polynomials_coefficients(hh.vec(), x, natural_monom_list)

    return [F0, F1, F2, F3, F4, AR, A, H1, H2], monom_list, remaining_monoms_list


def remove_monoms_from_remaining_stats(natural_statistics_symbolic: sp.MutableDenseMatrix,
                                       remaining_monom_list: list,
                                       dynamic_sde: SDE
                                       ):
    """
    Remove monomials from the remaining statistics that can be expressed as products of natural statistics.

    This function identifies and removes monomials from the remaining monomial list that can be expressed
    as products of the natural statistics. The expectations of these monomials are calculated using the
    Fisher metric, and the corresponding indices are stored for later use.

    Parameters
    ----------
    natural_statistics_symbolic : sp.MutableDenseMatrix
        Matrix of natural statistics in symbolic form.
    remaining_monom_list : list
        List of tuples where each tuple represents the degree of each variable in the monomial.
    dynamic_sde : SDE
        The stochastic differential equation (SDE) object containing the variables.

    Returns
    -------
    tuple
        A tuple containing:
        - higher_stats_indices_from_fisher : tuple
            Indices of the higher-order statistics that can be expressed as products of natural statistics.
        - updated_remaining_monom_list : list
            Updated list of remaining monomials after removing those that can be expressed as products of natural statistics.

    Notes
    -----
    This function is useful for projection filtering, where the remaining monomials are those that cannot
    be expressed as products of the natural statistics. The removed monomials' expectations are calculated
    using the Fisher metric, E[c_ic_j] = I[i,j] + E[c_i]*E[c_j].
    """
    higher_stats_indices_from_fisher_list = []
    monoms_tuples_to_be_removed_list = []
    c = natural_statistics_symbolic
    n_theta = len(c)
    for a_monom_degree in remaining_monom_list:
        a_monom = from_tuple_to_symbolic_monom(dynamic_sde.variables, a_monom_degree)
        for k in range(n_theta):
            for ell in range(k, n_theta):
                if a_monom == c[k] * c[ell]:
                    higher_stats_indices_from_fisher_list.append((k, ell))
                    monoms_tuples_to_be_removed_list.append(a_monom_degree)
                    # Break the inner loop...
                    break
            else:
                # Continue if the inner loop wasn't broken.
                continue
                # Inner loop was broken, break the outer.
            break

    # now that we have collected them, remove them from remaining_monom_list
    for a_monom_degree in monoms_tuples_to_be_removed_list:
        remaining_monom_list.remove(a_monom_degree)

    if higher_stats_indices_from_fisher_list:
        temp = jnp.array(higher_stats_indices_from_fisher_list)
        higher_stats_indices_from_fisher = (temp[:, 0], temp[:, 1])
    else:
        higher_stats_indices_from_fisher = ([], [])
    updated_remaining_monom_list = remaining_monom_list

    return higher_stats_indices_from_fisher, updated_remaining_monom_list


def construct_remaining_statistics(dynamic_sde: SDE,
                                   remaining_monom_list: list
                                   ):
    """
    Construct the remaining statistics from the given SDE and list of remaining monomials.

    This function generates the remaining statistics by converting each monomial degree in the
    remaining monomial list to its corresponding symbolic monomial form using the variables
    from the provided SDE.

    Parameters
    ----------
    dynamic_sde : SDE
        The stochastic differential equation (SDE) object containing the variables.
    remaining_monom_list : list
        List of tuples where each tuple represents the degree of each variable in the monomial.

    Returns
    -------
    sp.Matrix
        A matrix containing the remaining statistics in symbolic form, where each entry is a
        monomial formed by raising the variables to their corresponding degrees.

    Notes
    -----
    This function is useful for constructing the remaining statistics needed for projection
    filtering, where the remaining monomials are those that cannot be expressed as products
    of the natural statistics.
    """
    remaining_monoms = [from_tuple_to_symbolic_monom(dynamic_sde.variables, monomial_degree)
                        for monomial_degree in remaining_monom_list]
    remaining_statistics_symbolic = sp.Matrix(remaining_monoms)
    return remaining_statistics_symbolic


def natural_statistics_backward_kolmogorov_gen_fun_jax_generator(
        natural_statistics_symbolic: sp.Matrix,
        dynamic_sde: SDE,
        constant_substitution: dict,
        simplified: bool = True ) -> callable:
    """
    Generate a JAX function for the backward Kolmogorov generator of natural statistics.

    This function creates a JAX-compatible function that computes the backward Kolmogorov generator
    for a given set of natural statistics and a dynamic SDE. The resulting function can be used
    for efficient numerical computations.

    Parameters
    ----------
    natural_statistics_symbolic : sp.Matrix
        Matrix of natural statistics in symbolic form.
    dynamic_sde : SDE
        SDE object containing drift and diffusion terms of the dynamics.
    constant_substitution : dict
        Dictionary mapping symbolic constants to their numeric values.

    Returns
    -------
    Callable
        A JAX function that takes a state vector as input and returns the backward Kolmogorov generator
        evaluated at that state.

    Notes
    -----
    The returned function has the signature:
        f(x: ndarray[n_states]) -> ndarray[n_statistics]

    The function uses symbolic differentiation to compute the backward Kolmogorov generator and
    then converts the symbolic expression to a JAX-compatible function for efficient evaluation.
    """
    an_cal_l_c = backward_diffusion(natural_statistics_symbolic, dynamic_sde)
    if simplified:
        an_cal_l_c = sp.simplify(an_cal_l_c.subs(constant_substitution))
    an_cal_l_c_jax, _ = sympy_matrix_to_jax(an_cal_l_c,
                                            dynamic_sde.variables,
                                            disable_parameters=True)
    an_cal_l_c_jax = jnp.vectorize(an_cal_l_c_jax, signature='(n)->(m)')
    return an_cal_l_c_jax


def square_fwd_klmgr_p_div_p_jax(nat_stat_sym: sp.MutableDenseMatrix,
                                 dyn_sde: SDE,
                                 theta_sym: sp.MutableDenseMatrix,
                                 constant_substitution: dict):
    """
    Compute the squared term of the forward kolmogorov of p_theta the divide by p_theta (L^ast(p_theta)/p_theta).

    This function calculates (L^ast(p_theta)/p_theta))^2 where L^ast is the forward kolmogorov operator
    and p_theta is exponential family density with parameters theta.

    Parameters
    ----------
    nat_stat_sym : sp.MutableDenseMatrix
        Natural statistics/sufficient statistics in symbolic form
    dyn_sde : SDE
        SDE object containing drift and diffusion terms of the dynamics
    theta_sym : sp.MutableDenseMatrix
        Parameters of exponential family in symbolic form
    constant_substitution : dict
        Dictionary mapping symbolic constants to their numeric values

    Returns
    -------
    callable
        Vectorized JAX function that takes state x and parameters theta and returns
        squared term of forward kolmogorov of p/p_theta

    Notes
    -----
    The returned function has signature:
        f(x: ndarray[n_states], theta: ndarray[n_params]) -> float

    Uses the forward kolmogorov operator to compute (L^ast(p_theta)/p_theta))^2 where p_theta
    is exponential family density with parameters theta.
    """
    n_state = len(dyn_sde.variables)
    D = dyn_sde.diffusions * dyn_sde.diffusions.transpose()

    L_p_per_p = 0
    c_T_theta = (nat_stat_sym.transpose() * theta_sym)[0]

    for i in range(n_state):
        f_i = dyn_sde.drifts[i]
        x_i = dyn_sde.variables[i]
        L_p_per_p -= sp.diff(f_i, x_i) + f_i * sp.diff(c_T_theta, x_i)

        second_part = 0
        for j in range(n_state):
            x_j = dyn_sde.variables[j]
            D_ij = D[i, j]

            second_part += sp.diff(D_ij, x_j) + D_ij * sp.diff(c_T_theta, x_j)

        second_part = sp.diff(second_part, x_i) + second_part * sp.diff(c_T_theta, x_i)

        L_p_per_p += second_part

    L_p_per_p_squared = L_p_per_p ** 2 # no need for simplification
    result = L_p_per_p_squared.subs(constant_substitution)
    symbols_in = tuple((*dyn_sde.variables, *theta_sym.flat()))
    a_jax_fun, _ = sympy_matrix_to_jax(sp.Matrix([result]),
                                       symbols_in,
                                       squeeze=True,
                                       disable_parameters=True)

    #the returned function is decorated differently:
    def a_jax_fun_2(_x: Array, _theta: Array):
        augmented_arg = jnp.concatenate((_x, _theta))
        return a_jax_fun(augmented_arg)

    returned_jax_fun = jnp.vectorize(a_jax_fun_2, signature="(d)->()", excluded=(1,))

    return returned_jax_fun


def natural_statistics_backward_kolmogorov_numerical_jax(
        natural_statistics_fun: callable,
        drift_fun: callable,
        diffusion_fun: callable,
        t: float = 0.0,
        args=None
) -> callable:
    """
    Generate a JAX function for the backward Kolmogorov generator of natural statistics
    using automatic differentiation (numerical computation, no symbolic computation).

    This function computes L[c](x) where L is the backward Kolmogorov generator:
        L[c](x) = ∇c · f + (1/2) tr(Gᵀ H[c] G)

    Parameters
    ----------
    natural_statistics_fun : callable
        JAX function that evaluates natural statistics at state x.
        Signature: (x: Array[n_states]) -> Array[n_stats]
    drift_fun : callable
        Drift function of the SDE.
        Signature: (t: float, x: Array[n_states], args) -> Array[n_states]
    diffusion_fun : callable
        Diffusion matrix function of the SDE.
        Signature: (t: float, x: Array[n_states], args) -> Array[n_states, n_brownian]
    t : float, optional
        Time parameter for time-dependent drift/diffusion (default: 0.0)
    args : Any, optional
        Additional arguments passed to drift_fun and diffusion_fun

    Returns
    -------
    callable
        JAX function computing L[c](x) for all natural statistics.
        Signature: (x: Array[n_states]) -> Array[n_stats]

    Notes
    -----
    The backward Kolmogorov generator is:
        L[c](x) = Σᵢ (∂c/∂xᵢ) · fᵢ(x) + (1/2) Σᵢⱼ (GGᵀ)ᵢⱼ · (∂²c/∂xᵢ∂xⱼ)

    This implementation uses JAX automatic differentiation with JVP-based
    Hessian-vector products for efficiency. Instead of computing the full
    Hessian stack O(m × n²), it computes O(n_brownian) Hessian-vector products.

    The key identity used is:
        tr(Gᵀ H G) = Σ_α gα^T H gα
    where gα is the α-th column of G.
    """
    import jax

    def backward_kolmogorov_operator(x: Array) -> Array:
        # Evaluate drift and diffusion
        f = drift_fun(t, x, args)  # (n_states,)
        G = diffusion_fun(t, x, args)  # (n_states, n_brownian)

        # Jacobian: (n_stats, n_states)
        J = jax.jacobian(natural_statistics_fun)(x)
        drift_term = J @ f  # (n_stats,)

        # Diffusion term using JVP-based Hessian-vector products
        # tr(G^T H[c_k] G) = sum_alpha g_alpha^T H[c_k] g_alpha
        def compute_quadratic_form(g_alpha):
            # Compute g_alpha^T @ H[c_k] @ g_alpha for all k using JVP
            # H @ v = d/dε ∇c(x + εv)|_{ε=0}
            # jvp of jacobian gives us this
            _, Hv = jax.jvp(
                lambda x_: jax.jacobian(natural_statistics_fun)(x_),
                (x,), (g_alpha,)
            )
            # Hv has shape (n_stats, n_states)
            # Hv[k, i] = sum_j H[k,i,j] * g_alpha[j]
            # We need: sum_i g_alpha[i] * Hv[k, i] = g_alpha^T @ H[k] @ g_alpha
            return jnp.einsum('i,ki->k', g_alpha, Hv)

        # Vectorized over Brownian dimensions using vmap
        # G.T has shape (n_brownian, n_states), vmap over first axis
        all_quadratic_forms = jax.vmap(compute_quadratic_form)(G.T)  # (n_brownian, n_stats)
        diffusion_term = 0.5 * jnp.sum(all_quadratic_forms, axis=0)  # (n_stats,)

        return drift_term + diffusion_term

    return jnp.vectorize(backward_kolmogorov_operator, signature='(n)->(m)')
