from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import (
    Module, 
    ModuleList,
    Sequential,
    Conv2d,
    GroupNorm,
)

from neural_nets.conv_block import SeparableInvertResidual
from neural_nets.denoise_models.constituent_blocks import (
    DownBlock,
    UpBlock,
    SinusoidalPositionEmbedBlock,
    EmbedBlock, 
    LatentBlock
)

from neural_nets.activations import get_activation


class _UNetEncoder(Module):
    def __init__(
        self,
        img_channels: int,
        down_channels: Sequence[int],
        is_conditional: bool,
        T: int = 100, 
        t_embed_dim: int = 12,
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.T = T
        
        factory_kwargs = {"device": device, "dtype": dtype}
        num_stage = len(down_channels)
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation
        }
        
        #
        # main components
        #
        
        # backbone
        self.stem = Sequential(
            Conv2d(img_channels, down_channels[0], 3, padding=(1, 1), **factory_kwargs),
            get_activation(activation),
            SeparableInvertResidual(down_channels[0], down_channels[0], **invert_residual_kwargs, **factory_kwargs),
        )
        
        # add stages
        layers = []
        t_embeddings = []
        down_channels_per_stage = list(down_channels) 
        down_channels_per_stage = [down_channels[0]] + down_channels_per_stage
        for stage_i in range(0, num_stage):
            layers.append(
                DownBlock(
                    in_channels=down_channels_per_stage[stage_i],
                    out_channels=down_channels_per_stage[stage_i + 1],
                    is_conditional=is_conditional,
                    **kwargs,
                    **invert_residual_kwargs,
                    **factory_kwargs
                )
            )
            t_embeddings.append(
                EmbedBlock(
                    input_dim=t_embed_dim, 
                    embed_dim=down_channels_per_stage[stage_i], 
                    activation=activation, 
                    **factory_kwargs
                )
            )
            
        self.latent = LatentBlock(
            latent_dim=down_channels[-1],
            expand_factor=expand_factor,
            num_group_norm=num_groups_norm,
            activation=activation,
            **factory_kwargs
        )
        
        self.layers = ModuleList(layers)
        self.t_embeddings = ModuleList(t_embeddings)
        
        # additional components
        self.sinusoidal_time = SinusoidalPositionEmbedBlock(t_embed_dim)
    
    
    def forward(
        self, 
        input: Tensor,
        time: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        
        Z = self.stem(input) # stem block forward    
        t = time.float() / self.T # normalize the time
        t = self.sinusoidal_time(t) # init time embedding
        
        encoder_outputs = [Z]
        encoder_outputs.append(Z)
        
        for i, stage in enumerate(self.layers):
            t_i = self.t_embeddings[i](t) # learn time embedding
            Z = stage(Z + t_i, *args, **kwargs)
            encoder_outputs.append(Z)
            
        return self.latent(Z), tuple(encoder_outputs)
        
    
class _UNetDecoder(Module):
    def __init__(
        self,
        img_channels: int,
        up_channels: Sequence[int],
        is_conditional: bool,
        T: int = 100, 
        t_embed_dim: int = 12,
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        **kwargs
    ):
        super().__init__()
        self.T = T
        
        factory_kwargs = {"device": device, "dtype": dtype}
        num_stage = len(up_channels)
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation
        }
        
        #
        # main components
        #
        
        # add stages
        layers = []
        t_embeddings = []
        up_channels_per_stage = list(up_channels) 
        up_channels_per_stage = up_channels_per_stage + [up_channels_per_stage[-1]]
        
        for stage_i in range(0, num_stage):
            layers.append(
                UpBlock(
                    in_channels=up_channels_per_stage[stage_i],
                    out_channels=up_channels_per_stage[stage_i + 1],
                    is_conditional=is_conditional,
                    **kwargs,
                    **invert_residual_kwargs,
                    **factory_kwargs
                )
            )
            t_embeddings.append(
                EmbedBlock(
                    input_dim=t_embed_dim, 
                    embed_dim=up_channels_per_stage[stage_i], 
                    activation=activation, 
                    **factory_kwargs
                )
            )
            
        # out layer
        self.out = Sequential(
            # concat with stem output
            Conv2d(2*up_channels[-1], up_channels[-1], 1, **factory_kwargs),
            get_activation(activation),
            GroupNorm(num_groups_norm, up_channels[-1], **factory_kwargs),
            
            # learn the spatial properties
            Conv2d(
                up_channels[-1], up_channels[-1], 
                kernel_size=3, 
                padding=(1, 1), 
                groups=up_channels[-1], 
                **factory_kwargs
            ),
            get_activation(activation),
            GroupNorm(num_groups_norm, up_channels[-1], **factory_kwargs),
            
            SeparableInvertResidual(
                up_channels[-1], 
                up_channels[-1], 
                expand_factor, 
                num_groups_norm=num_groups_norm,
                activation=activation,
                **factory_kwargs
            ),
            
            SeparableInvertResidual(
                up_channels[-1], 
                up_channels[-1], 
                expand_factor, 
                num_groups_norm=num_groups_norm,
                activation=activation,
                **factory_kwargs
            ),
            
            Conv2d(up_channels[-1], img_channels, kernel_size=1, **factory_kwargs),
        )
        
        self.layers = ModuleList(layers)
        self.t_embeddings = ModuleList(t_embeddings)
        
        # additional components
        self.sinusoidal_time = SinusoidalPositionEmbedBlock(t_embed_dim)
    
    
    def forward(
        self, 
        input: Tensor,
        encoder_outputs: Tuple[Tensor],
        time: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
    
        Z = input
        memory = encoder_outputs[::-1]
        t = time.float() / self.T # normalize the time
        t = self.sinusoidal_time(t) # init time embedding
        
        for i, stage in enumerate(self.layers):
            t_i = self.t_embeddings[i](t) # learn time embedding
            Z = stage(Z + t_i, memory[i], *args, **kwargs)
            
        # project to the new image space (noise space)
        return self.out(torch.cat([Z, memory[-1]], 1))


class _UNetBase(Module):
    def __init__(
        self, 
        img_channels: int,
        down_channels: Sequence[int],
        is_conditional: bool,
        T: int = 100,  
        t_embed_dim: int = 12,
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        up_channels = down_channels[::-1] # reverse of the down channels
        self.T = T
        
        #
        # main components
        #
        
        # backbone
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation
        }
        
        self.encoder = _UNetEncoder(
            img_channels=img_channels,
            down_channels=down_channels,
            is_conditional=is_conditional,
            **invert_residual_kwargs,
            **factory_kwargs,
            **kwargs,
        )
        
        self.decoder = _UNetDecoder(
            img_channels=img_channels,
            up_channels=up_channels,
            is_conditional=is_conditional,
            T=T,
            t_embed_dim=t_embed_dim,
            **invert_residual_kwargs,
            **factory_kwargs,
            **kwargs,
        )
        
        
    def forward(
        self, 
        input: Tensor,
        time: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        r"""
        Forward method of predicting noise UNet

        Args:
            input (Tensor): Noisy input
            time (Tensor): Time step t

        Shape:
            input: [N, C, H, W]
            time: [1]
        
        Returns:
            Tensor [N, C, H, W]:  the predicted noise that previously added to the input
        """        
        # encoder forward pass
        Z, encoder_outputs = self.encoder(input, time, *args, **kwargs)
        return self.decoder(Z, encoder_outputs, time, *args, **kwargs)
        
 
class UNet(_UNetBase):
    def __init__(self, *arg, **kwargs):
        super().__init__(is_conditional=False, *arg, **kwargs)
        
        
class UNetConditional(_UNetBase):
    def __init__(self, *arg, **kwargs):
        super().__init__(is_conditional=True, *arg, **kwargs)
