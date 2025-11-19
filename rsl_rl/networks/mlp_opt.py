from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import resolve_nn_activation

from rsl_rl.networks import MLP, Optimization


class MlpOpt(nn.Module):
    """Multi-layer perceptron with an optimisation layer in the tail.

    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        n_envs: int,
        ns: int,
        nx: int,
        nu: int,
        dt: float,
        w: float,
        ub: list[int],
        lb: list[int],
        policy_hidden_dims: list[int],
        policy_activation: str = "elu",
        policy_last_activation: str = "softplus",
    ) -> None:
        """Initialize the MlpOpt.

        Args:
            TODO: Add args description
        """
        super().__init__()

        self.ns = ns
        self.nx = nx
        self.nu = nu
        self.nvars = self.ns * self.nx + (self.ns - 1) * self.nu
        policy_input_dim = self.nx + self.nvars
        policy_output_dim = 2 * self.nx + self.nu
        mpc_input_dim = policy_input_dim + policy_output_dim

        ## Cost policy
        self.cost_policy = MLP(
            input_dim=policy_input_dim,
            output_dim=policy_output_dim,
            hidden_dims=policy_hidden_dims,
            activation=policy_activation,
            last_activation=policy_last_activation,
        )

        ## Optimization Layer
        self.opt_layer = Optimization(
            n_batch=n_envs,
            ns=self.ns,
            nx=self.nx,
            nu=self.nu,
            dt=dt,
            w=w,
            ub=ub,
            lb=lb,
        )

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        self.cost_policy.init_weights(scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor.
        """
        x_init = x[:, 0 : self.nx].clone()
        z_des = x[:, self.nx :].clone()
        # Get MPC costs from policy
        x = self.cost_policy.forward(x)
        # Pass through optimization layer
        x = self.opt_layer.forward(x, x_init, z_des)
        return x
