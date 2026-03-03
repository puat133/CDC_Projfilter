import itertools
from collections.abc import Callable
import jax.scipy.special as jsp
import jax.numpy as jnp
import sympy as sp

# Special since need to reduce arguments.
MUL = 0
ADD = 1

_jnp_func_lookup = {
    sp.Mul: MUL,
    sp.Add: ADD,
    sp.div: "jnp.div",
    sp.Abs: "jnp.abs",
    sp.sign: "jnp.sign",
    # Note: May raise error for ints.
    sp.ceiling: "jnp.ceil",
    sp.floor: "jnp.floor",
    sp.log: "jnp.log",
    sp.exp: "jnp.exp",
    sp.sqrt: "jnp.sqrt",
    sp.cos: "jnp.cos",
    sp.acos: "jnp.acos",
    sp.sin: "jnp.sin",
    sp.asin: "jnp.asin",
    sp.tan: "jnp.tan",
    sp.atan: "jnp.arctan",
    sp.atan2: "jnp.arctan2",
    # Note: Also may give NaN for complex results.
    sp.cosh: "jnp.cosh",
    sp.acosh: "jnp.acosh",
    sp.sinh: "jnp.sinh",
    sp.asinh: "jnp.asinh",
    sp.tanh: "jnp.tanh",
    sp.atanh: "jnp.atanh",
    sp.Pow: "jnp.power",
    sp.re: "jnp.real",
    sp.im: "jnp.imag",
    # Note: May raise error for ints and complexes
    sp.erf: "jsp.erf",
    sp.erfc: "jsp.erfc",
    sp.LessThan: "jnp.less",
    sp.GreaterThan: "jnp.greater",
    sp.And: "jnp.logical_and",
    sp.Or: "jnp.logical_or",
    sp.Not: "jnp.logical_not",
    sp.Max: "jnp.fmax",
    sp.Min: "jnp.fmin",
    sp.Mod: "jnp.fmod",
}


def sympy_matrix_to_jax(sympy_matrix: sp.MutableDenseMatrix,
                        symbols_in: sp.Symbol,
                        retain_shape: bool =False,
                        squeeze: bool=False,
                        disable_parameters: bool = False):
    """Converts a sympy expression into a function with jax operations.

    Parameters
    ----------
    sympy_matrix : sp.MutableDenseMatrix
        Input sympy matrix to convert
    symbols_in : sp.Symbol
        Input symbols used in sympy expressions
    retain_shape : bool, optional
        If True, reshapes output to match input matrix shape. Default is False
    squeeze : bool, optional
        If True, squeezes output array. Default is False
    disable_parameters : bool, optional
        If True, disables parameter extraction. Default is False

    Returns
    -------
    tuple
        (function, parameters) where:
        - function executes the jax operations
        - parameters are extracted constants from expressions
    """
    #   name a function
    hash_string = 'A_' + str(abs(hash(str(sympy_matrix) + str(symbols_in))))

    #   set initial parameters to empty list
    parameters = []

    #   flatten the matrix expression
    sympy_flatten_list = list(itertools.chain.from_iterable(sympy_matrix.tolist()))

    functional_form_text = ""
    for expression in sympy_flatten_list:
        functional_form_text += sympy2jaxtext(expression, parameters, symbols_in, disable_parameters)
        functional_form_text += ","
    # remove last comma
    functional_form_text = functional_form_text[:-1]

    if len(parameters) > 0:
        text = f"def {hash_string}(X, parameters):\n"
    else:
        text = f"def {hash_string}(X):\n"

    text += "    return jnp.array(["
    text += functional_form_text

    # add closed bracket and reshaping
    text += "])"
    if retain_shape:
        text += ".reshape(({},{}))".format(sympy_matrix.shape[0], sympy_matrix.shape[1])
    if squeeze:
        text += ".squeeze()"

    ldict = {}
    exec(text, globals(), ldict)
    return ldict[hash_string], jnp.array(parameters)


def sympy2jaxtext(expr, parameters, symbols_in, disable_parameters=False):
    """Converts a sympy expression into jax operations.

    Parameters
    ----------
    expr : sp.Expr
        Input sympy expression to convert
    parameters : list
        List to store extracted parameters from expressions
    symbols_in : sp.Symbol
        Input symbols used in sympy expressions
    disable_parameters : bool, optional
        If True, disables parameter extraction. Default is False

    Returns
    -------
    str
        String containing jax operations translating the input expression
    """
    if issubclass(expr.func, sp.Float):
        if disable_parameters:
            # print("Parameter disabled!")
            return f"{float(expr)}"
        else:
            parameters.append(float(expr))
            return f"parameters[{len(parameters) - 1}]"
    elif issubclass(expr.func, sp.Integer):
        return f"{int(expr)}"
    elif issubclass(expr.func, sp.Rational):
        # print("There is a fraction here {}".format(expr))
        res = float(int(expr.p)/int(expr.q))
        return f"{res}"
    elif issubclass(expr.func, sp.Symbol):
        matching_symbols = [i for i in range(len(symbols_in)) if symbols_in[i] == expr]
        if len(matching_symbols) == 0:
            raise ValueError(f"The expression symbol {expr} was not found in user-passed `symbols_in`: {symbols_in}.")
        elif len(matching_symbols) > 1:
            raise ValueError(
                f"The expression symbol {expr} was found more than once in user-passed `symbols_in`: {symbols_in}.")
        if len(symbols_in) > 1:
            return f"X[{matching_symbols[0]}]"
        else:
            return f"X"
    else:
        try:
            _func = _jnp_func_lookup[expr.func]
        except KeyError:
            raise KeyError("Function not found in sympy2jax.sympy2jax._jnp_func_lookup; please add it.")
        args = [sympy2jaxtext(arg, parameters, symbols_in, disable_parameters) for arg in expr.args]
        if _func == MUL:
            return ' * '.join(['(' + arg + ')' for arg in args])
        elif _func == ADD:
            return ' + '.join(['(' + arg + ')' for arg in args])
        else:
            return f'{_func}({", ".join(args)})'


def sympy2jax(equation, symbols_in, disable_parameters=False):
    """Converts a sympy expression into a jax operation.

    Parameters
    ----------
    equation : sp.Expr
        Input sympy expression to convert
    symbols_in : sp.Symbol
        Input symbols used in sympy expressions
    disable_parameters : bool, optional
        If True, disables parameter extraction. Default is False

    Returns
    -------
    tuple
        (function, parameters) where:
        - function executes the jax operations
        - parameters are extracted constants from expressions
    """


    parameters = []
    functional_form_text = sympy2jaxtext(equation, parameters, symbols_in, disable_parameters)
    hash_string = 'A_' + str(abs(hash(str(equation) + str(symbols_in))))
    if len(parameters) > 0:
        text = f"def {hash_string}(X, parameters):\n"
    else:
        text = f"def {hash_string}(X):\n"
    text += "    return "
    text += functional_form_text
    ldict = {}
    exec(text, globals(), ldict)
    return ldict[hash_string], jnp.array(parameters)


def lamdify(vector_expression: sp.MutableDenseMatrix) -> Callable:
    """Converts a sympy matrix expression into a callable numpy function.

    A wrapper around sympy's lambdify function that converts sympy matrix expressions
    into callable functions that use numpy operations.

    Parameters
    ----------
    vector_expression : sp.MutableDenseMatrix
        Input sympy matrix expression to convert to a callable function

    Returns
    -------
    Callable
        A function that takes the free symbols as arguments and returns numpy array output

    Notes
    -----
    This function extracts all free symbols from the expression and uses them as the function arguments.
    The output will use numpy operations to evaluate the expressions.

    Examples
    --------
    >>> x, y = sp.symbols('x y')
    >>> expr = sp.MutableDenseMatrix([[x**2, y], [x+y, y**2]])
    >>> f = lamdify(expr)
    >>> f(2.0, 3.0)
    array([[4., 3.],
           [5., 9.]])
    """
    return sp.lambdify(list(vector_expression.free_symbols), vector_expression, 'numpy')
