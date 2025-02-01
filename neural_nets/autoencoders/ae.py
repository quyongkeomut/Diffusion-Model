from typing import (
    Sequence,
    Optional,
    List,
    Any
)
    
import torch
from torch import Tensor
from torch.nn import (
    Module, 
    Sequential,
    Conv2d,
    GroupNorm,
)

from torchvision.models.mobilenetv3 import (
    MobileNetV3, 
    _mobilenet_v3_conf, 
    MobileNet_V3_Large_Weights, 
    _ovewrite_named_param,
    InvertedResidualConfig
)
from torchvision.models._api import (
    WeightsEnum,
)

from neural_nets.conv_block import SeparableInvertResidual
from neural_nets.activations import get_activation

from neural_nets.autoencoders.constituent_blocks import DownBlock, UpBlock


class Encoder(Module):
    def __init__(
        self,
        img_channels: int,
        down_channels: Sequence[int],
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        
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
        layers = [
            # stem
            Sequential(
                Conv2d(
                    img_channels, 
                    down_channels[0],
                    kernel_size=1,
                    **factory_kwargs
                ),
                get_activation(activation),
            )
        ]
        
        # add stages
        down_channels_per_stage = list(down_channels) 
        down_channels_per_stage = [down_channels[0]] + down_channels_per_stage
        for stage_i in range(0, num_stage):
            layers.append(
                DownBlock(
                    in_channels=down_channels_per_stage[stage_i],
                    out_channels=down_channels_per_stage[stage_i + 1],
                    **invert_residual_kwargs,
                    **factory_kwargs
                )
            )
        
        # latent layer
        layers.append(
            SeparableInvertResidual(
                down_channels_per_stage[stage_i + 1], 
                down_channels_per_stage[stage_i + 1], 
                **invert_residual_kwargs, **factory_kwargs),
        )
        
        self.layers = Sequential(*layers)
    
    
    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)
    
   
class Decoder(Module):
    def __init__(
        self,
        img_channels: int,
        latent_channels: int,
        up_channels: Sequence[int],
        expand_factor: int = 3,
        num_groups_norm: int = 4,
        activation: str = "hardswish",
        device=None,
        dtype=None,
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        num_stage = len(up_channels)
        invert_residual_kwargs = {
            "expand_factor": expand_factor, 
            "num_groups_norm": num_groups_norm, 
            "activation": activation
        }
        self.img_channels = img_channels
        self.latent_channels = latent_channels
        
        #
        # main components
        #
        
        # add stages
        layers = []
        projection = [
            Conv2d(latent_channels, up_channels[0], 1, **factory_kwargs),
            GroupNorm(num_groups_norm, up_channels[0], **factory_kwargs)
        ]
        layers.extend(projection)
        
        up_channels_per_stage = list(up_channels) 
        up_channels_per_stage = up_channels_per_stage + [up_channels_per_stage[-1]]
        for stage_i in range(0, num_stage):
            layers.append(UpBlock(up_channels[stage_i], up_channels_per_stage[stage_i + 1], **invert_residual_kwargs, **factory_kwargs))
        
        # out layer
        layers.append(
            Sequential(
                Conv2d(
                    up_channels[-1], up_channels[-1], 3, 
                    padding=(1, 1), 
                    groups=up_channels[-1], 
                    **factory_kwargs
                ),
                get_activation(activation),
                Conv2d(up_channels[-1], img_channels, kernel_size=1, **factory_kwargs),
                get_activation("sigmoid"),
            )
        )
        
        self.layers = Sequential(*layers)
    
    
    def forward(
        self, 
        input: Tensor,
    ) -> Tensor:
        return self.layers(input)
    
    
class MobileNetV3_backbone(MobileNetV3):
    def __init__(self, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.eval()
        self.requires_grad_(False)
        if kwargs["is_domain_encoder"]:
            self.depth = 7
        else:
            self.depth = 4
        
    def forward(self, input: Tensor):
        # Z = input
        # for i in range(self.depth): # total strides = 4, number of channels = 24 OR total strides = 8, channels = 40
        #     Z = self.features[i](Z)
        # return Z
        return self.features[:self.depth](input)
    
    
def _mobilenet_v3_backbone(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional[WeightsEnum],
    is_domain_encoder: bool,
    **kwargs: Any,
) -> MobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = MobileNetV3_backbone(inverted_residual_setting, last_channel, is_domain_encoder=is_domain_encoder, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=False, check_hash=True))
    
    return model


def encoder_mobilenet_v3_large(
    *, weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2, is_domain_encoder: bool = False, **kwargs: Any
) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V3_Large_Weights` below for
            more details, and possible values.
        **kwargs: parameters passed to the ``torchvision.models.mobilenet.MobileNetV3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V3_Large_Weights
        :members:
    """
    weights = MobileNet_V3_Large_Weights.verify(weights)

    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return _mobilenet_v3_backbone(inverted_residual_setting, last_channel, weights, is_domain_encoder, **kwargs)
    