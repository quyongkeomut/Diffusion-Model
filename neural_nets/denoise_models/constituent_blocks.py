import math

import torch
from torch import (
    Tensor,
    transpose,
    flatten,
    unflatten
)

from torch.nn import (
    Module, 
    ModuleList,
    Unflatten,
    Sequential,
    Conv2d,
    Linear,
    GroupNorm,
    Unflatten,
    Upsample,
    MultiheadAttention,
)

from neural_nets.conv_block import SeparableInvertResidual

from neural_nets.activations import get_activation


class ConditionalModule(Module):
    def __init__(
        self, 
        in_channels: int,
        num_heads: int,
        activation: str,
        depth: int = 1,
        dropout: float = 0.1,
        device=None,
        dtype=None,
        **kwargs, # kdim, vdim
    ):
        r"""
        Conditional module is the wrapper of cross-attention layer

        Args:
            in_channels (int): number of channels of current representation
            num_heads (int): number of attention head
            dropout (float, optional): Dropout value used in attention matrix. 
                Defaults to 0.1.
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        # cross-attention
        self.mhas = [
            MultiheadAttention(
                embed_dim=in_channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
                **kwargs, # kdim, vdim
                **factory_kwargs
                ) 
            for i in range(depth)
        ]
        self.mhas = ModuleList(self.mhas)
        
        self.mha_norms = [
            GroupNorm(
                num_groups=num_heads,
                num_channels=in_channels,
                **factory_kwargs
            )
            for i in range(depth)
        ]
        self.mha_norms = ModuleList(self.mha_norms)
        
        # feed_forward
        self.linears = [
            Linear(in_channels, in_channels, **factory_kwargs)
            for i in range(depth)
        ]        
        self.linears = ModuleList(self.linears)
        
        self.linear_norms = [
            GroupNorm(
                num_groups=num_heads,
                num_channels=in_channels,
                **factory_kwargs
            )
            for i in range(depth)
        ]
        self.linear_norms = ModuleList(self.linear_norms)
        
        self.activations = [get_activation(activation) for i in range(depth)]
        self.activations = ModuleList(self.activations)
            

    def forward(
        self,
        querry: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        r"""
        Forward method of conditional module

        Args:
            querry (Tensor): Current representation, flattened
            condition (Tensor): key and value in multihead attention, which is the representation
                for conditional generative model
        
        Shape:
            querry: [N, H*W, C]
                
        Returns:
            Tensor: Output of cross-attention
        """
        Z = querry # [N, H*W, C]
        for i in range(len(self.mhas)):
            # cross-attention
            Z_input = Z.transpose(-1, -2).unsqueeze(-1).contiguous() # [N, C, H*W, 1]
            Z: Tensor = self.mhas[i](Z, condition, condition, **kwargs)[0] # [N, H*W, C]
            Z = Z.transpose(-1, -2).unsqueeze(-1).contiguous() # [N, C, H*W, 1]
            Z = self.mha_norms[i](Z + Z_input) # residual connection # [N, C, H*W, 1]
            
            # feed-forward
            Z_input = Z # [N, C, H*W, 1]
            Z: Tensor = self.linears[i](Z_input.squeeze(-1).transpose(-1, -2)) # [N, H*W, C]
            Z = self.activations[i](Z) # [N, H*W, C]
            Z = self.linear_norms[i](Z.transpose(-1, -2).unsqueeze(-1) + Z_input) # [N, C, H*W, 1]
            Z = Z.squeeze(-1).transpose(-1, -2) # [N, H*W, C]
            
        return Z
    

class DownBlock(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        is_conditional: bool,
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        **kwargs,
    ):  
        r"""
        Downsample block of lightweight UNet

        Args:
            in_channels (int): input channels
            out_channels (int): output channels
            is_conditional (bool): determine if this module is for conditional 
                generator or not. 
            expand_factor (int, optional): Expand factor used in expansion conv layer
                of inverted residual block . Defaults to 3.
            num_groups_norm (int, optional): Number of group to be normalized by group norm. 
                Defaults to 4.
            activation (str, optional): Activation function. Defaults to "hardswish".
        """
        super().__init__()
        
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation
        }
        factory_kwargs = {"device": device, "dtype": dtype}
        self.is_conditional = is_conditional
        
        layers = [
            Conv2d(
                in_channels, 
                in_channels,
                kernel_size=3,
                stride=2,
                padding=(1, 1),
                groups=in_channels,
                **factory_kwargs
            ),
            get_activation(activation),
            Conv2d(
                in_channels, 
                out_channels,
                kernel_size=1,
                **factory_kwargs
            ),
            get_activation(activation),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
        ]
        if is_conditional:
            for pos in [5, 7, 9]:
                conditional_module = ConditionalModule(
                    out_channels,
                    num_heads=num_groups_norm,
                    activation=activation,
                    dropout=0.1,
                    **factory_kwargs,
                    **kwargs, # kdim, vdim
                )
                layers.insert(pos, conditional_module)
            
        self.layers = ModuleList(layers)


    def forward(
        self, 
        input: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        r"""
        Forward method of Down block

        Args:
            input (Tensor): Input representation

            kwargs: key-word argument for multihead attention layer
            
        Returns:
            Tensor: Output
        """
        Z = input
        for i, layer in enumerate(self.layers):
            if i in [5, 7, 9] and self.is_conditional:
                H, W = Z.shape[2:]
                Z = flatten(Z, start_dim=2) # [N, C, H*W]
                Z = transpose(Z, 1, 2).contiguous() # [N, H*W, C]
                Z = layer(Z, *args, **kwargs)
                Z = transpose(Z, 1, 2).contiguous() # [N, C, H*W]
                Z = unflatten(Z, -1, (H, W)) # [N, C, H, W]
            else:
                Z = layer(Z)
        
        return Z    
        

class UpBlock(Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        is_conditional: bool,
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        **kwargs
    ):  
        super().__init__()
        
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation
        }
        factory_kwargs = {"device": device, "dtype": dtype}
        self.is_conditional = is_conditional
        
        layers = [
            Conv2d(
                2*in_channels, 
                out_channels,
                kernel_size=1,
                **factory_kwargs
            ),
            get_activation(activation),
            Upsample(scale_factor=2, mode="bilinear"),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
        ]
        if is_conditional:
            for pos in [4, 6, 8]:
                conditional_module = ConditionalModule(
                    out_channels,
                    num_heads=num_groups_norm,
                    activation=activation,
                    dropout=0.1,
                    **factory_kwargs,
                    **kwargs, # kdim, vdim
                )
                layers.insert(pos, conditional_module)
            
        self.layers = ModuleList(layers)


    def forward(
        self, 
        input: Tensor, 
        memory: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        Z = torch.cat([input, memory], dim=1)
        
        for i, layer in enumerate(self.layers):
            if i in [4, 6, 8] and self.is_conditional:
                H, W = Z.shape[2:]
                Z = flatten(Z, start_dim=2) # [N, C, H*W]
                Z = transpose(Z, 1, 2).contiguous() # [N, H*W, C]
                Z = layer(Z, *args, **kwargs)
                Z = transpose(Z, 1, 2).contiguous() # [N, C, H*W]
                Z = unflatten(Z, -1, (H, W)) # [N, C, H, W]
            else:
                Z = layer(Z)
        
        return Z


class SinusoidalPositionEmbedBlock(Module):
    def __init__(self, dim: int):
        r"""
        This module generate sinusoidal positional embedding for time embedding.

        Args:
            dim (int): _description_
        """
        super().__init__()
        self.dim = dim


    def forward(self, time: Tensor) -> Tensor:
        r"""
        Returns time embedding based on the time step t

        Args:
            time (Tensor): Time at step t

        Shape:
            time: [1]
        
        Returns:
            Tensor: Time embedding based on the time step t
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class EmbedBlock(Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int,
        activation: str = "hardswish",
        device=None,
        dtype=None
    ):
        r"""
        This block is the encoder for additional embeddings, such as time embedding and
        positional embedding for the Denoising model.        

        Args:
            input_dim (int): number of dimension of additional representation.
            emb_dim (int): number of dimension of embedding, equals number of channels
                feature map.
            activation (str, optional): Nonlinear activation. Defaults to "hardswish".
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.input_dim = input_dim
        layers = [
            Linear(input_dim, embed_dim, **factory_kwargs),
            get_activation(activation),
            Linear(embed_dim, embed_dim, **factory_kwargs),
            Unflatten(1, (embed_dim, 1, 1)),
        ]
        self.layers = Sequential(*layers)


    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)


class LatentBlock(Module):
    def __init__(
        self, 
        latent_dim: int,
        expand_factor: int,
        num_group_norm: int,
        activation: str = "hardswish", 
        device=None,
        dtype=None
    ):
        r"""
        Latent tensor learning block. The input tensor is flattened, then
        fed into linear layers followed by nonlinear activation.

        Args:
            latent_dim (int): Dimension of latent representation
            latent_image_shape (Tuple[int]): Shape of the image before getting
            flattened
            activation (str, optional): Nonlinear activation. Defaults to "hardswish".
        """
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        layers = [
            SeparableInvertResidual(latent_dim, latent_dim, expand_factor, 1, num_group_norm, activation, **factory_kwargs)
            for i in range(3) 
        ]
        self.layers = Sequential(*layers)
        
        
    def forward(self, input: Tensor) -> Tensor:
        # C, H, W = input.shape[1:]
        # Z = input.flatten(1) # [N, C*H*W]
        # Z: Tensor = self.layers(Z)
        # Z = Z.unflatten(-1, (C, H, W))
        # return Z
        return self.layers(input)
        
    