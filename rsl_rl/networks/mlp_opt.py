from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import resolve_nn_activation

from rsl_rl.networks import MLP, ConstrainedLqr


class MlpOpt(nn.Module):
    """Multi-layer perceptron with an optimisation layer in the tail.

    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        ns: int,
        nx: int,
        nu: int,
        input_dim: int,
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
        mlp_output_dim = int(self.nx / 2) + self.nu + self.nx
        mpc_input_dim = policy_input_dim + mlp_output_dim

        ## Cost policy
        self.cost_policy = MLP(
            input_dim=input_dim,
            output_dim=mlp_output_dim,
            hidden_dims=policy_hidden_dims,
            activation=policy_activation,
            last_activation=policy_last_activation,
        )

        ## Optimization Layer
        self.opt_layer = ConstrainedLqr(ns=self.ns, nx=self.nx, nu=self.nu)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """
        self.cost_policy.init_weights(scales)

    def forward(self, x: torch.Tensor, A, b, G, h) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor.
        """
        # Get MPC costs from policy
        x = self.cost_policy.forward(x)
        # Pass through optimization layer
        x = self.opt_layer.forward(x, A, b, G, h)
        return x
