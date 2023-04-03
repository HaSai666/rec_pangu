# -*- ecoding: utf-8 -*-
# @ModuleName: deep
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
from typing import List, Union
import torch.nn as nn
from .activation import get_activation


class MLP(nn.Module):
    """Customizable Multi-Layer Perceptron"""

    def __init__(self,
                 input_dim: int,
                 output_dim: Union[int, None] = None,
                 hidden_units: List[int] = [],
                 hidden_activations: Union[str, List[str]] = "ReLU",
                 output_activation: Union[str, None] = None,
                 dropout_rates: Union[float, List[float]] = 0.1,
                 batch_norm: bool = False,
                 use_bias: bool = True):
        """Initialize the MLP layer.
        Args:
            input_dim: Size of the input layer.
            output_dim: Size of the output layer (optional).
            hidden_units: List of hidden layer sizes.
            hidden_activations: Activation function for each hidden layer.
            output_activation: Activation function for the output layer (optional).
            dropout_rates: Dropout rate for the hidden layers (same for all if float,
                            otherwise list for individual layers).
            batch_norm: If True, apply batch normalization to each hidden layer.
            use_bias: If True, use bias in each dense layer.
        """
        super(MLP, self).__init__()

        # Error checking:
        if output_dim is not None:
            assert isinstance(output_dim, int) and output_dim > 0, "output_dim must be an integer"
        assert isinstance(input_dim, int) and input_dim > 0, "input_dim must be an integer"
        assert isinstance(hidden_units, list) and all(isinstance(i, int) for i in hidden_units) and len(
            hidden_units) >= 1, "hidden_units must be a list of integers and with at least one element"
        if isinstance(hidden_activations, str):
            hidden_activations = [hidden_activations] * len(hidden_units)
        elif isinstance(hidden_activations, list):
            assert len(hidden_activations) == len(
                hidden_units), "hidden_activations must have one element per hidden unit"
        else:
            raise TypeError("hidden_activations must be a string or a list of strings")

        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        elif isinstance(dropout_rates, list):
            assert len(dropout_rates) == len(hidden_units), "dropout_rates must have one element per hidden unit"
        else:
            raise TypeError("dropout_rates must be a float or a list of floats")

        # Prepend input dim to hidden layer sizes list
        hidden_units = [input_dim] + hidden_units
        dense_layers = []
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(get_activation(hidden_activations[idx]))
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))

        self.net = nn.Sequential(*dense_layers)

    def forward(self, x):
        """Forward propagate through the neural network.
        Args:
            x: Input tensor with shape (batch_size, input_dim).
        Returns:
            Output tensor with shape (batch_size, output_dim) if output_dim is not None,
            otherwise tensor with shape (batch_size, hidden_units[-1]).
        """
        return self.net(x)
