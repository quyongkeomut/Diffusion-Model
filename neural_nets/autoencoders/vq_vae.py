from typing import Tuple
    
import torch
from torch import randn, Tensor
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    ModuleList,
    Embedding
)

from neural_nets.activations import get_activation
from neural_nets.autoencoders.base_ae import Encoder, Decoder, LATENT_H, LATENT_W

from utils.initializers import ones_
from utils.other_utils import to_image
    
    
class VQ_VAEEncoder(Encoder):
    ...
    
    
class VQ_VAEDecoder(Decoder):
    def __init__(
        self, 
        codebook_size: int,
        codebook_dim: int,
        retriever_depth: int = 3,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        factory_kwargs = {"device": kwargs["device"], "dtype": kwargs["dtype"]}
        self.codebook = Embedding(
            num_embeddings=codebook_size,
            embedding_dim=codebook_dim
        ) # Parameter(randn(codebook_size, codebook_dim, **factory_kwargs))
    
        dim_feedforward=2*codebook_dim
        self.retriever = ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=codebook_dim,
                    nhead=kwargs["nhead"],
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    **factory_kwargs,
                )
                for _ in range(retriever_depth)
            ]
        )

    def forward(self, input: Tensor) -> Tensor:
        ...
    
    def sample(
        self,
        num_samples: int,
    ):
        ...
    
    def generate(self):
        ...