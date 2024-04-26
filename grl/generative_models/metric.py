import time
from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict
from torch.distributions import Distribution, MultivariateNormal

from grl.generative_models.diffusion_model import (
    DiffusionModel, EnergyConditionalDiffusionModel)
from grl.numerical_methods.numerical_solvers.ode_solver import (
    DictTensorODESolver, ODESolver)
from grl.numerical_methods.ode import ODE


def compute_likelihood(
        model: Union[DiffusionModel, EnergyConditionalDiffusionModel],
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        t: torch.Tensor = None,
        condition: Union[torch.Tensor, TensorDict] = None,
        using_Hutchinson_trace_estimator: bool = True,
    ) -> torch.Tensor:
    """
    Overview:
        Compute Likelihood of samples in generative model for gaussian prior.
    Arguments:
        - model (:obj:`Union[Callable, nn.Module]`): The model.
        - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state.
        - t (:obj:`torch.Tensor`): The input time.
        - condition (:obj:`Union[torch.Tensor, TensorDict]`): The input condition.
        - using_Hutchinson_trace_estimator (:obj:`bool`): Whether to use Hutchinson trace estimator. It is an approximation of the trace of the Jacobian of the drift function, which is faster but less accurate. We recommend setting it to True for high dimensional data.
    Returns:
        - log_likelihood (:obj:`torch.Tensor`): The likelihood of the samples.
    """
    #TODO: Add support for EnergyConditionalDiffusionModel; Add support for t; Add support for treetensor.torch.Tensor

    model_drift = model.diffusion_process.forward_ode(function=model.model, function_type=model.model_type, condition=condition).drift

    def compute_trace_of_jacobian_general(dx, x):
        # Assuming x has shape (B, D1, ..., Dn)
        shape = x.shape[1:]  # get the shape of a single element in the batch
        outputs = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # trace for each batch
        # Iterate through each index in the product of dimensions
        for index in torch.cartesian_prod(*(torch.arange(s) for s in shape)):
            if len(index.shape)>0:
                index = tuple(index)
            else:
                index = (index,)
            grad_outputs = torch.zeros_like(x)
            grad_outputs[(slice(None), *index)] = 1  # set one at the specific index across all batches
            grads = torch.autograd.grad(outputs=dx, inputs=x, grad_outputs=grad_outputs, retain_graph=True)[0]
            outputs += grads[(slice(None), *index)]
        return outputs


    def compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x, eps):
        """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

        fn_eps = torch.sum(dx * eps)
        grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
        outputs = torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
        return outputs

    def composite_drift(t, x):
        # where x is actually x0_and_diff_logp, which is a dict containing x and logp_xt_minus_logp_x0
        with torch.set_grad_enabled(True):
            t = t.detach()
            x_t = x['x'].detach()
            logp_xt_minus_logp_x0 = x['logp_xt_minus_logp_x0']

            x_t.requires_grad = True
            t.requires_grad = True

            dx = model_drift(t, x_t)
            if using_Hutchinson_trace_estimator:
                noise = torch.randn_like(x_t, device=x_t.device)
                logp_drift = - compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x_t, noise)
                # logp_drift = - divergence_approx(dx, x_t, noise)
            else:
                logp_drift = - compute_trace_of_jacobian_general(dx, x_t)

            delta_x = treetensor.torch.tensor({'x': dx, 'logp_xt_minus_logp_x0': logp_drift}, device=x_t.device)
            return delta_x

    # x.shape = [batch_size, state_dim]
    x0_and_diff_logp = treetensor.torch.tensor({'x': x, 'logp_xt_minus_logp_x0': torch.zeros(x.shape[0])}, device=x.device)

    eps = 1e-3
    t_span = torch.linspace(eps, 1.0, 1000).to(x.device)


    solver = DictTensorODESolver(library="torchdyn", dict_type="treetensor")

    x1_and_logp1 = solver.integrate(
        drift=composite_drift,
        x0=x0_and_diff_logp,
        t_span=t_span,
        batch_size=x.shape[0],
        x_size=x0_and_diff_logp.shape
    )[-1]

    logp_x1_minus_logp_x0 = x1_and_logp1['logp_xt_minus_logp_x0']
    x1 = x1_and_logp1['x']
    x1_1d = x1.reshape(x1.shape[0], -1)
    logp_x1 = MultivariateNormal(loc=torch.zeros_like(x1_1d, device=x1_1d.device), covariance_matrix=torch.eye(x1_1d.shape[-1], device=x1_1d.device)).log_prob(x1_1d)

    log_likelihood = logp_x1 - logp_x1_minus_logp_x0

    return log_likelihood
