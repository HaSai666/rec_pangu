# -*- ecoding: utf-8 -*-
# @ModuleName: activation
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2022/6/10 7:40 PM
import torch
from torch import nn


class Dice(nn.Module):
    """Implements the Dice activation function.

    Args:
        input_dim (int): dimensionality of the input tensor
        eps (float, optional): term added to the denominator to provide numerical stability (default: 1e-9)
    """

    def __init__(self, input_dim: int, eps: float = 1e-9):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=eps, momentum=0.01)
        self.alpha = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs forward pass of the Dice activation function.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            output (torch.Tensor): output tensor after applying the Dice activation function
        """
        p = torch.sigmoid(self.bn(x))
        output = p * x + (1 - p) * self.alpha * x
        return output


def get_activation(activation: str or nn.Module) -> nn.Module:
    """Returns the PyTorch activation function object corresponding to a string.

    Args:
        activation (str or nn.Module): name of the activation function or PyTorch activation function object

    Returns:
        activation_fn (nn.Module): PyTorch activation function object
    """
    if isinstance(activation, str):
        activation_str = activation.lower()
        if activation_str == "relu":
            activation_fn = nn.ReLU()
        elif activation_str == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation_str == "tanh":
            activation_fn = nn.Tanh()
        else:
            activation_fn = getattr(nn, activation)()
    else:
        activation_fn = activation

    return activation_fn
