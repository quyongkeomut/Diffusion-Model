from typing import Callable
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
    BatchNorm1d,
    BatchNorm2d,
    Unflatten,
    ConvTranspose2d,
    MultiheadAttention,
)

from neural_nets.conv_block import SeparableInvertResidual
from neural_nets.activations import get_activation
from utils.initializers import _get_initializer, ones_
from neural_nets.denoise_models.dm_utils import ConstituentBlockHolder


class GenericBlock(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.argholder = ConstituentBlockHolder(*args, **kwargs)
        self.initializer = _get_initializer(self.argholder.initializer)
        self._set_backbone()
        
        
    def _set_backbone(self):
        raise NotImplementedError
    
    
    def _reset_parameters(self):
        raise NotImplementedError


class DownBlock(GenericBlock):
    def _set_backbone(self):
        invert_residual_kwargs = {
            "expand_factor": self.argholder.expand_factor, 
            "drop_p": self.argholder.drop_p,
            "base_groups_norm": self.argholder.base_groups_norm,
            "activation": self.argholder.activation,
            "initializer": self.initializer
        }
        factory_kwargs = {"device": self.argholder.device, "dtype": self.argholder.dtype}
        conv_kwargs = {"kernel_size": 3, "padding": (1, 1), "bias": True}
        in_channels, out_channels = self.argholder.in_channels, self.argholder.out_channels
        base_groups_norm = self.argholder.base_groups_norm
        
        layers = [
            # conv for downsampling 
            Sequential(
                Conv2d(in_channels, out_channels, stride=2, **conv_kwargs, **factory_kwargs),
                # BatchNorm2d(out_channels, **factory_kwargs),
                GroupNorm(
                    num_groups=4*base_groups_norm,
                    num_channels=out_channels,
                    **factory_kwargs
                ),
                get_activation(self.argholder.activation, **factory_kwargs),
            ),
            # 1
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            # 3
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            #
            Sequential(
                Conv2d(out_channels, out_channels, **conv_kwargs, **factory_kwargs),
                # BatchNorm2d(out_channels, **factory_kwargs),
                GroupNorm(
                    num_groups=4*base_groups_norm,
                    num_channels=out_channels,
                    **factory_kwargs
                ),
                get_activation(self.argholder.activation, **factory_kwargs),
            )
        ]
        
        if self.argholder.is_conditional:
            # conditional_module = ConditionalModule(
            #     out_channels,
            #     activation=self.activation,
            #     drop_p=self.argholder.drop_p,
            #     self.argholder.kdim, 
            #     self.argholder.vdim
            #     factory_kwargs
            # )
            # layers.insert(1, conditional_module)
            ...
            
        self.layers = ModuleList(layers)
        DownBlock._reset_parameters(self)


    def _reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, Conv2d):    
                        self.initializer(sublayer.weight)
                        if sublayer.bias is not None:
                            ones_(sublayer.bias)


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
        for layer in self.layers:
            if isinstance(layer, ConditionalModule):
                H, W = Z.shape[2:]
                Z = flatten(Z, start_dim=2) # [N, C, H*W]
                Z = transpose(Z, 1, 2).contiguous() # [N, H*W, C]
                Z = layer(Z, *args, **kwargs)
                Z = transpose(Z, 1, 2).contiguous() # [N, C, H*W]
                Z = unflatten(Z, -1, (H, W)) # [N, C, H, W]
            else:
                Z = layer(Z)
        
        return Z    
        

class UpBlock(GenericBlock):
    def _set_backbone(self):
        invert_residual_kwargs = {
            "expand_factor": self.argholder.expand_factor, 
            "drop_p": self.argholder.drop_p,
            "base_groups_norm": self.argholder.base_groups_norm,
            "activation": self.argholder.activation,
            "initializer": self.initializer
        }
        factory_kwargs = {"device": self.argholder.device, "dtype": self.argholder.dtype}
        in_channels, out_channels = self.argholder.in_channels, self.argholder.out_channels
        base_groups_norm = self.argholder.base_groups_norm
        conv_k, conv_stride, conv_pad =  3, 1, (1, 1)
        convt_k, convt_stride, convt_pad = 2, 2, (0, 0)
        
        layers = [          
            Sequential(
                Conv2d(2*in_channels, in_channels, conv_k, conv_stride, conv_pad, bias=True, **factory_kwargs),
                # BatchNorm2d(in_channels, **factory_kwargs),
                GroupNorm(
                    num_groups=4*base_groups_norm,
                    num_channels=in_channels,
                    **factory_kwargs
                ),
                get_activation(self.argholder.activation, **factory_kwargs),
            ),
            # 1
            SeparableInvertResidual(in_channels, in_channels, **invert_residual_kwargs, **factory_kwargs),
            # 3            
            Sequential(
                ConvTranspose2d(in_channels, out_channels, convt_k, convt_stride, convt_pad, bias=True, **factory_kwargs),
                # BatchNorm2d(out_channels, **factory_kwargs),
                GroupNorm(
                    num_groups=4*base_groups_norm,
                    num_channels=out_channels,
                    **factory_kwargs
                ),
                get_activation(self.argholder.activation, **factory_kwargs),
            ),
            # 5
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
            # 7
            SeparableInvertResidual(out_channels, out_channels, **invert_residual_kwargs, **factory_kwargs),
        ]
        if self.argholder.is_conditional:
            # conditional_module = ConditionalModule(
            #     in_channels,
            #     num_heads=base_groups_norm,
            #     activation=activation,
            #     dropout=0.1,
            #     **factory_kwargs,
            #     **kwargs, # kdim, vdim
            # )
            # layers.insert(1, conditional_module)
            pass
            
        self.layers = ModuleList(layers)
        UpBlock._reset_parameters(self)
        

    def _reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, (Conv2d, ConvTranspose2d)):    
                        self.initializer(sublayer.weight)
                        if sublayer.bias is not None:
                            ones_(sublayer.bias)


    def forward(
        self, 
        input: Tensor, 
        encoder_output: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        Z = input
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                Z = torch.cat([Z, encoder_output], dim=1) # (N, 2*C, H, W)
                Z = layer(Z)
            elif isinstance(layer, ConditionalModule):
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
        self.half_dim = self.dim // 2
        self.log_10_pow_4 = math.log(10000)
        self.pre_embeddings = self.log_10_pow_4 / (self.half_dim - 1)


    def forward(self, time: Tensor) -> Tensor:
        r"""
        Returns time embedding based on the time step t

        Args:
            time (Tensor): Time at step t

        Shape:
            time: (N, )
        
        Returns:
            Tensor (N, t_embed_dim): Time embedding based on the time step t
        """
        device = time.device
        time = time.view(time.shape[0], 1) # (N, 1)
        embeddings = torch.exp(torch.arange(self.half_dim, device=device) * -self.pre_embeddings) # (half_dim)
        embeddings = time * embeddings[None, :] # (N, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # (N, dim)
        return embeddings


class EmbedBlock(Module):
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int,
        activation: str = "hardswish",
        initializer: str | Callable[[Tensor], Tensor] = "he_uniform",
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
        self.initializer = _get_initializer(initializer)
        layers = [
            Linear(input_dim, 2*embed_dim, bias=True, **factory_kwargs),
            # BatchNorm1d(2*embed_dim, **factory_kwargs),
            get_activation(activation, **factory_kwargs),
            
            Linear(2*embed_dim, embed_dim, **factory_kwargs),
            
            Unflatten(1, (embed_dim, 1, 1)),
        ]
        self.layers = Sequential(*layers)
        self._reset_parameters()
        
        
    def _reset_parameters(self):
        self.initializer(self.layers[0].weight)
        self.initializer(self.layers[2].weight)
        ones_(self.layers[2].bias)


    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
    
class ConditionalModule(Module):
    def __init__(
        self, 
        in_channels: int,
        activation: str,
        depth,
        drop_p: float,
        kdim,
        vdim,
        device=None,
        dtype=None,
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
                num_heads=8,
                dropout=drop_p,
                batch_first=True,
                kdim=kdim, 
                vdim=vdim,
                **factory_kwargs
                ) 
            for i in range(depth)
        ]
        self.mhas = ModuleList(self.mhas)
        
        self.mha_norms = [
            GroupNorm(
                num_groups=1,
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
                num_groups=1,
                num_channels=in_channels,
                **factory_kwargs
            )
            for i in range(depth)
        ]
        self.linear_norms = ModuleList(self.linear_norms)
        
        self.activations = [get_activation(activation, **factory_kwargs) for i in range(depth)]
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
    