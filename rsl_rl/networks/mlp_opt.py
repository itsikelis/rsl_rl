from __future__ import annotations

import torch
import torch.nn as nn
from functools import reduce

from rsl_rl.utils import resolve_nn_activation

from rsl_rl.modules.opt_layer import OptLayer


class MlpOpt(nn.Sequential):
    """Multi-layer perceptron with an optimisation layer in the tail.

    The MlpOpt network is a sequence of linear layers and activation functions. The last layer is a differentiable optimisation layer that solves a QP prblem.

    It provides additional conveniences:
    - If the hidden dimensions have a value of ``-1``, the dimension is inferred from the input dimension.
    - If the output dimension is a tuple, the output is reshaped to the desired shape.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int] | list[int],
        activation: str = "elu",
        last_activation: str = "softplus",
        ns: int = 10,
        nx: int = 4,
        nu: int = 2,
    ) -> None:
        """Initialize the MlpOpt.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output.
            hidden_dims: Dimensions of the hidden layers. A value of ``-1`` indicates that the dimension should be
                inferred from the input dimension.
            activation: Activation function.
            last_activation: Activation function of the last cost policy layer.
        """
        super().__init__()
        nvars = nx + nu + nx  # intermediate stage costs + final
        policy_output_dim = 2 * nx + nu
        mpc_input_dim = input_dim + policy_output_dim

        # Resolve activation functions
        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation)
        # Resolve number of hidden dims if they are -1
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(
                nn.Linear(
                    hidden_dims_processed[layer_index],
                    hidden_dims_processed[layer_index + 1],
                )
            )
            layers.append(activation_mod)

        # Add last cost policy layer
        if isinstance(policy_output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], policy_output_dim))
        else:
            print("Error: Non scalar output dimension not supported in MlpOpt class!")
            exit()

        # Add last activation function
        layers.append(last_activation_mod)

        ## Add optimisation function in the back ##
        opt_layer = OptLayer()
        layers.append(opt_layer)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize the weights of the MLP.

        Args:
            scales: Scale factor for the weights.
        """

        def get_scale(idx: int) -> float:
            """Get the scale factor for the weights of the MLP.

            Args:
                idx: Index of the layer.
            """
            return scales[idx] if isinstance(scales, (list, tuple)) else scales

        # Initialize the weights
        for idx, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_scale(idx))
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP.

        Args:
            x: Input tensor.
        """
        for layer in self:
            x = layer(x)
        return x
