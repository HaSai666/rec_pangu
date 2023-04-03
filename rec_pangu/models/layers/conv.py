# -*- ecoding: utf-8 -*-
# @ModuleName: conv
# @Author: wk
# @Email: 306178200@qq.com
# @Time: 2023/4/3 15:15
from typing import List
from torch import nn
import torch


class NextItNetLayer(nn.Module):
    def __init__(self, channels: int, dilations: List[int], one_masked: bool, kernel_size: int, feat_drop: float = 0.0):
        """
        Args:
            channels: Number of input channels
            dilations: List of dilation sizes for each residual block
            one_masked: Whether to use one-mask convolutions
            kernel_size: Size of convolutional kernel
            feat_drop: Dropout probability for input features, default 0.0
        """
        super().__init__()
        if one_masked:
            ResBlock = ResBlockOneMasked
            if dilations is None:
                dilations = [1, 2, 4]
        else:
            ResBlock = ResBlockTwoMasked
            if dilations is None:
                dilations = [1, 4]
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop > 0 else None
        self.res_blocks = nn.ModuleList([
            ResBlock(channels, kernel_size, dilation) for dilation in dilations
        ])

    def forward(self, emb_seqs, lens):
        """
        Args:
            emb_seqs: Input sequence of embeddings, shape (batch_size, max_len, channels)
            lens: Length of sequences in batch, shape (batch_size,)

        Returns:
            The final state tensor, shape (batch_size, channels)
        """
        batch_size, max_len, _ = emb_seqs.size()
        mask = torch.arange(
            max_len, device=lens.device
        ).unsqueeze(0).expand(batch_size, max_len) >= lens.unsqueeze(-1)
        emb_seqs = torch.masked_fill(emb_seqs, mask.unsqueeze(-1), 0)
        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)

        x = torch.transpose(emb_seqs, 1, 2)  # (B, C, L)
        for res_block in self.res_blocks:
            x = res_block(x)
        batch_idx = torch.arange(len(lens))
        last_idx = lens - 1
        sr = x[batch_idx, :, last_idx]  # (B, C)
        return sr


class ResBlockOneMasked(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        """
        Initialize a ResBlockOneMasked object.

        Args:
        - channels (int): the number of input channels.
        - kernel_size (int): the size of the convolutional kernel.
        - dilation (int): the dilation factor of the convolutional kernel.
        """
        super().__init__()
        mid_channels = channels // 2
        self.layer_norm1 = LayerNorm(channels)
        self.conv1 = nn.Conv1d(channels, mid_channels, kernel_size=1)
        self.layer_norm2 = LayerNorm(mid_channels)
        self.conv2 = MaskedConv1d(
            mid_channels, mid_channels, kernel_size=kernel_size, dilation=dilation
        )
        self.layer_norm3 = LayerNorm(mid_channels)
        self.conv3 = nn.Conv1d(mid_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResBlockOneMasked object.

        Args:
        - x (torch.Tensor): input tensor with size (B, C, L).

        Returns:
        - y (torch.Tensor): output tensor with size (B, C, L).
        """
        y = x
        y = torch.relu(self.layer_norm1(y))
        y = self.conv1(y)
        y = torch.relu(self.layer_norm2(y))
        y = self.conv2(y)
        y = torch.relu(self.layer_norm3(y))
        y = self.conv3(y)
        return y + x


class MaskedConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        """
        This class implements a 1d convolutional neural network.

        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - kernel_size (int): Size of the kernel/window.
        - dilation (int): Controls the spacing between the kernel points. Default is 1.

        Returns:
        - None
        """
        super().__init__()

        # Creates a 1D convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

        # Calculates amount of padding
        self.padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolves the input tensor using the parameters set during initialization.

        Args:
        - x (torch.Tensor): Input tensor with size (B,C,L), where B is the batch size,
        C is the number of channels, and L is the length of the sequence.

        Returns:
        - x (torch.Tensor): Output tensor from the convolutional operation of size (B, out_channels, L + padding),
        where out_channels is the number of output channels and L is the length of the sequence.
        """
        # Add padding to the input tensor.
        x = torch.nn.functional.pad(x, [self.padding, 0])  # (B, C, L + self.padding)

        # Apply the convolutional operation.
        x = self.conv(x)

        return x


class LayerNorm(nn.Module):
    """
    Layer normalization operation.
    Args:
        channels (int): Number of channels in the input tensor.
        epsilon (float, optional): Small number to avoid numerical instability.
    """

    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        """
        Initialize layer normalization.
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones([1, channels, 1], dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros([1, channels, 1], dtype=torch.float32))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of layer normalization.
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, num_channels, sequence_length).
        Returns:
            The tensor normalized by layer norm operation.
        """
        # Calculate variance and mean of the input tensor along the channel dimension.
        var, mean = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
        # Calculate the normalization term.
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        # Apply the scale (gamma) and shift (beta) parameters to the normalized tensor.
        return x * self.gamma + self.beta


class ResBlockTwoMasked(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        """
        A residual block with two masked convolutions and layer normalization.

        Args:
            channels (int): Number of channels
            kernel_size (int): Size of the convolving kernel
            dilation (int): Spacing between kernel elements
        """
        super().__init__()
        self.conv1 = MaskedConv1d(channels, channels, kernel_size, dilation)
        self.layer_norm1 = LayerNorm(channels)
        self.conv2 = MaskedConv1d(channels, channels, kernel_size, 2 * dilation)
        self.layer_norm2 = LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            y (torch.Tensor): Output tensor
        """
        y = x
        y = self.conv1(y)
        y = torch.relu(self.layer_norm1(y))
        y = self.conv2(y)
        y = torch.relu(self.layer_norm2(y))
        return y + x
