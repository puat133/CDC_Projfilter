from functools import partial
from typing import TypeVar
import cd_filtering.bayesian_update as bu
import diffrax as dfx
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax._custom_types import Control, VF, Y
from equinox import filter_jit
from jaxtyping import Array

_VF = TypeVar("_VF", bound=VF)
_Control = TypeVar("_Control", bound=Control)


@filter_jit
@partial(jnp.vectorize, signature='(m,n),(n)->(m)')
def _mat_vec_product(a_matrix: Array, a_vector: Array) -> Array:
    return a_matrix @ a_vector


class VectorizedControlTerm(dfx.ControlTerm):
    def prod(self, vf: _VF, control: _Control) -> Y:
        # return jtu.tree_map(_prod, vf, control)
        return jtu.tree_map(_mat_vec_product, vf, control)


class RiemannianSteadyStateEvent(dfx.SteadyStateEvent):
    def __call__(self, state, *, terms, args, solver, stepsize_controller, **kwargs):
        del kwargs
        msg = (
            "The `rtol` and `atol` tolerances for `SteadyStateEvent` default "
            "to the `rtol` and `atol` used with an adaptive step size "
            "controller (such as `diffrax.PIDController`). Either use an "
            "adaptive step size controller, or specify these tolerances "
            "manually."
        )
        if self.rtol is None:
            if isinstance(stepsize_controller, dfx.AbstractAdaptiveStepSizeController):
                _rtol = stepsize_controller.rtol
            else:
                raise ValueError(msg)
        else:
            _rtol = self.rtol
        if self.atol is None:
            if isinstance(stepsize_controller, dfx.AbstractAdaptiveStepSizeController):
                _atol = stepsize_controller.atol
            else:
                raise ValueError(msg)
        else:
            _atol = self.atol

        # TODO: this makes an additional function evaluation that in practice has
        # probably already been made by the solver.
        vf = solver.func(terms, state.tprev, state.y, args)

        # We only care about the riemannian norm of d_theta
        theta, mu_params, cov_params, learn_rate = state.y
        params = (mu_params, cov_params, 1.)
        fisher = terms.term.vector_field.p.fisher_metric(theta, params)
        riem_norm_vf = bu.riemannian_norm(vf[0],fisher)

        return riem_norm_vf < _atol
