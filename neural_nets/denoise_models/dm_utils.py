from typing import Callable, Sequence

from torch import Tensor
    
class ArgumentHolder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
            
class ConstituentBlockHolder(ArgumentHolder):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        is_conditional: bool,
        expand_factor: int = 4,
        drop_p: float=0.1,
        activation: str = "gelu",
        initializer: str | Callable[[Tensor], Tensor] = "normal",
        device=None,
        dtype=None,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            is_conditional=is_conditional,
            expand_factor=expand_factor,
            drop_p=drop_p,
            activation=activation,
            initializer=initializer,
            device=device,
            dtype=dtype,
            **kwargs
        )
        
        
class UNetHolder(ArgumentHolder):
    def __init__(
        self,
        img_channels: int,
        stages_channels: Sequence[int],
        latent_dim: int,
        is_conditional: bool,
        T: int = 100, 
        t_embed_dim: int = 32,
        **kwargs
    ):
        super().__init__(
            img_channels=img_channels,
            stages_channels=stages_channels,
            latent_dim=latent_dim,
            is_conditional=is_conditional,
            T=T, 
            t_embed_dim=t_embed_dim,
            **kwargs
        )