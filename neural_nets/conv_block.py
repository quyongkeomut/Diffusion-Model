import torch
from torch.nn import (
    Module,
    Sequential,
    Conv2d,
    GroupNorm
)

from neural_nets.activations import get_activation


class SeparableInvertResidual(Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        expand_factor: int = 3,
        stride: int = 1,
        num_groups_norm: int = 4,
        activation: str = "swish",
        device=None,
        dtype=None
    ):
        r"""
        Implementation of Inverted Residual block, which is introduced in MobileNetv2.

        Args:
            in_channels (int): number of channels of input.
            out_channels (int): number of channels of output.
            expand_factor (int, optional): Expand factor of the first pointwise conv. Defaults to 3.
            num_groups_norm (int, optional): Number of group for GN layer. Defaults to 4.
            activation (str, optional): Name of nonlinear activation function. Defaults to "swish".
        """
        super().__init__() 
        
        factory_kwargs = {"device": device, "dtype": dtype}
        expand_channels = expand_factor*in_channels
        if in_channels == out_channels and stride == 1:
            self.is_residual = True
        else:
            self.is_residual = False
        
        # pointwise
        expansion = [
            Conv2d(
               in_channels=in_channels, 
               out_channels=expand_channels,
               kernel_size=1,
               **factory_kwargs
            ),
            get_activation(activation),
            GroupNorm(
                num_groups=num_groups_norm,
                num_channels=expand_channels,
                **factory_kwargs
            ),
        ]
        self.expansion = Sequential(*expansion)
        
        # depthwise
        depthwise = [
            Conv2d(
                in_channels=expand_channels, 
                out_channels=expand_channels,
                kernel_size=3,
                stride=stride,
                padding=(1, 1),
                groups=expand_channels,
                **factory_kwargs
            ),
            get_activation(activation),
            GroupNorm(
                num_groups=num_groups_norm,
                num_channels=expand_channels,
                **factory_kwargs
            ),
        ]
        self.depthwise = Sequential(*depthwise)
        
        # pointwise
        projection = [
            Conv2d(
                in_channels=expand_channels, 
                out_channels=out_channels,
                kernel_size=1,
                **factory_kwargs
            ),
            GroupNorm(
                num_groups=num_groups_norm,
                num_channels=out_channels,
                **factory_kwargs
            ),
        ]
        self.projection = Sequential(*projection)
        
        
    def forward(self, input):
        Z = self.projection(self.depthwise(self.expansion(input)))
        if self.is_residual:
            Z = Z + input
        return Z