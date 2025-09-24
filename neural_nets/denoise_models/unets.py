from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    Module, 
    ModuleList,
    Sequential,
    Conv2d,
    BatchNorm2d,
    GroupNorm,
    Tanh,
)

from neural_nets.denoise_models.constituent_blocks import (
    DownBlock,
    UpBlock,
    SinusoidalPositionEmbedBlock,
    EmbedBlock, 
)

from neural_nets.activations import get_activation
from utils.initializers import _get_initializer, ones_, zeros_
from neural_nets.denoise_models.dm_utils import UNetHolder


LATENT_H, LATENT_W = 1, 1


class BaseUNetEncoder(Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.argholder = UNetHolder(**kwargs)
        self.downblock_kwargs = kwargs
        self.initializer = _get_initializer(self.argholder.initializer)
        self.T = self.argholder.T
        self._set_backbone()
        
    
    def _set_backbone(self):  
        activation = self.argholder.activation
        stages_channels = self.argholder.stages_channels
        t_embed_dim = self.argholder.t_embed_dim
        base_groups_norm = self.argholder.base_groups_norm
        factory_kwargs = {"device": self.argholder.device, "dtype": self.argholder.dtype}
        conv_kwargs = {"kernel_size": 3, "padding": (1, 1)}
        
        # BACKBONE     
        self.stem_layers = [
            Conv2d(self.argholder.img_channels, stages_channels[0], **conv_kwargs, **factory_kwargs),
            get_activation(activation, **factory_kwargs),
            # projection
        ]
        
        layers = []
        t_embeddings = []
        for idx, stage_i in enumerate(stages_channels):
            in_channels = stages_channels[0] if idx == 0 else stages_channels[idx-1]
            layers.append(
                DownBlock(in_channels=in_channels, out_channels=stage_i, **self.downblock_kwargs)
            )
            t_embeddings.append(
                EmbedBlock(
                    input_dim=t_embed_dim, 
                    embed_dim=in_channels, 
                    activation=activation, 
                    initializer=self.initializer,
                    **factory_kwargs
                )
            )
        
        self.stem_layers = Sequential(*self.stem_layers)
        self.layers = ModuleList(layers)
        self.t_embeddings = ModuleList(t_embeddings)
        
        latent_dim = self.argholder.latent_dim
        self.latent_layers = Sequential(
            Conv2d(stage_i, latent_dim, stride=1, bias=True, **conv_kwargs, **factory_kwargs),
            # BatchNorm2d(latent_dim, **factory_kwargs),
            GroupNorm(
                num_groups=8*base_groups_norm,
                num_channels=latent_dim,
                **factory_kwargs
            ),
            get_activation(activation, **factory_kwargs),
        )
        
        # additional components
        self.sinusoidal_time = SinusoidalPositionEmbedBlock(t_embed_dim)
        BaseUNetEncoder._reset_parameters(self)
    
    
    def _reset_parameters(self):
        self.initializer(self.stem_layers[0].weight)
        ones_(self.stem_layers[0].bias)
        self.initializer(self.latent_layers[0].weight)
        ones_(self.latent_layers[0].bias)
    
    
    def forward(
        self, 
        input: Tensor,
        time: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tuple[Tensor]]:
        """
        Forward method of UNet Encoder

        Args:
            input (Tensor): Input of Encoder
            time (Tensor): Time at step t
            
        Shape:
            input: (N, C, H, W)
            time: (N, )

        Returns:
            Tuple[Tensor, Tuple[Tensor]]: Latent representation and outputs from stages
        """
        t = time.float() / self.T # normalize the time
        t = self.sinusoidal_time(t) # init time embedding; shape (N, t_embed_dim)
        Z = self.stem_layers(input)
        encoder_outputs = []
        # encoder_outputs.append(Z)
        
        for i, stage in enumerate(self.layers):
            Z: Tensor = stage(Z + self.t_embeddings[i](t), *args, **kwargs) 
            encoder_outputs.append(Z)
            
        latent = self.latent_layers(Z)
        return latent, tuple(encoder_outputs)
        
    
class BaseUNetDecoder(Module):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs["stages_channels"] = kwargs["stages_channels"][::-1] 
        self.argholder = UNetHolder(**kwargs)
        self.upblock_kwargs = kwargs
        self.initializer = _get_initializer(self.argholder.initializer)
        self.T = self.argholder.T
        self._set_backbone()
        
        
    def _set_backbone(self):   
        latent_dim = self.argholder.latent_dim
        t_embed_dim = self.argholder.t_embed_dim
        activation = self.argholder.activation
        base_groups_norm = self.argholder.base_groups_norm
        factory_kwargs = {"device": self.argholder.device, "dtype": self.argholder.dtype}
        conv_kwargs = {"kernel_size": 3, "padding": (1, 1)}
        stages_channels = self.argholder.stages_channels
        num_stages = len(stages_channels)
        
        # BACKBONE
        layers = []
        t_embeddings = []
        decoder_input_layers = [
            Conv2d(latent_dim, stages_channels[0], bias=True, **conv_kwargs, **factory_kwargs),
            # BatchNorm2d(stages_channels[0], **factory_kwargs),
            GroupNorm(
                num_groups=4*base_groups_norm,
                num_channels=stages_channels[0],
                **factory_kwargs
            ),
            get_activation(activation, **factory_kwargs),
        ]  # learn the latent
        
        for idx, stage_i in enumerate(stages_channels):
            out_channels = stages_channels[idx + 1] if idx != num_stages - 1 else stage_i
            layers.append(  
                UpBlock(in_channels=stage_i, out_channels=out_channels, **self.upblock_kwargs)
            )
            t_embeddings.append(EmbedBlock(t_embed_dim, stage_i, activation, **factory_kwargs))
            
        # out layer
        self.out_layers = [
            Conv2d(stage_i, stage_i, **conv_kwargs, **factory_kwargs),
            get_activation(activation, **factory_kwargs),
            # projection
            Conv2d(stage_i, self.argholder.img_channels, **conv_kwargs, **factory_kwargs),
            Tanh(),
        ]
        
        self.decoder_input_layers = Sequential(*decoder_input_layers)
        self.layers = ModuleList(layers)
        self.t_embeddings = ModuleList(t_embeddings)
        self.out_layers = Sequential(*self.out_layers)
        
        # additional components
        self.sinusoidal_time = SinusoidalPositionEmbedBlock(t_embed_dim)
        BaseUNetDecoder._reset_parameters(self)
    
    
    def _reset_parameters(self):
        for module in [self.decoder_input_layers, self.out_layers]:
            for layer in module:
                if isinstance(layer, Conv2d):
                    self.initializer(layer.weight)
                    if layer.bias is not None:
                        ones_(layer.bias)
        zeros_(self.out_layers[-2].bias)
        
    
    def forward(
        self, 
        input: Tensor,
        encoder_outputs: Tuple[Tensor],
        time: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """
        Forward method of UNet Decoder

        Args:
            input (Tensor): Latent from the Encoder
            encoder_outputs (Tuple[Tensor]): Outputs from stages of Encoder
            time (Tensor): Time at step t
            
        Shape:
            input: (N, C, H, W)
            time: (N, )

        Returns:
            Tensor: Mask output of Decoder
        """
    
        Z = self.decoder_input_layers(input)
        
        encoder_outputs = encoder_outputs[::-1]
        t = time.float() / self.T # normalize the time
        t = self.sinusoidal_time(t) # init time embedding; shape (N, t_embed_dim)
        
        for i, stage in enumerate(self.layers):
            Z = stage(Z + self.t_embeddings[i](t), encoder_outputs[i], *args, **kwargs)
            
        # project to the image space (noise space)
        return self.out_layers(Z)


class BaseUNet(Module):
    def __init__(self, **kwargs,):
        super().__init__()
        self.encoder = BaseUNetEncoder(**kwargs)      
        self.decoder = BaseUNetDecoder(**kwargs)
        
        
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
            input: (N, C, H, W)
            time: (N, )
        
        Returns:
            Tensor (N, C, H, W):  The output mask 
        """        
        # encoder forward pass
        Z, encoder_outputs = self.encoder(input, time, *args, **kwargs)
        return self.decoder(Z, encoder_outputs, time, *args, **kwargs)