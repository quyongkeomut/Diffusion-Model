from typing import (
    Any,
    Tuple
)

import torch
from torch import Tensor

from torch.nn import (Module)
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

class ELBOLoss(Module):
    def __init__(
        self, 
        reconstruction_method: str = "mse",
        use_elbo: bool = True,
        eps: float = 1e-4,
        *args, 
        **kwargs
    ) -> None:
        """
        Implementation of ELBO (Evidence Lower-Bound) for DDPM
        
        Args:
            eps (float, optional): Smoothing value. Defaults to 1e-5.
        """
        super().__init__()
        self.eps = eps
        assert reconstruction_method in ["mse", "bce"], (
            f"reconstruction_method must be one of ['mse', 'bce'], got {reconstruction_method} instead."
        )
        self.use_elbo = use_elbo
        self.reconstruction_method = mse_loss if reconstruction_method == "mse" else binary_cross_entropy_with_logits
    
    
    def forward(
        self, 
        ddpm: Module,
        T: Tensor,
        X_pred: Tensor,
        X_input: Tensor,
        **kwargs,
    ) -> Tensor:
        """
        ELBO Loss Forward method

        Args:
            ddpm (Module): 
            T (Tensor): 
            X_pred (Tensor): 
            X_input (Tensor): Input data

        Shape:
            T: (N, )
            X_pred: (N, C, H, W)
            X_input: (N, C, H, W)
        
        Returns:
            Tensor: Scalar value of ELBO Loss
        """
        if self.use_elbo:
            if 'is_ddp' in kwargs.keys() and kwargs['is_ddp'] is True:
                vars_T  = (ddpm.module.sqrt_one_minus_alphas_bar[T-1] * ddpm.module.bw_std[T])**2 # (N, 1, 1, 1)
                weight = 0.5 * ddpm.module.square_betas[T] * ddpm.module.alphas_bar[T-1] / (vars_T * ddpm.module.square_one_minus_alphas_bar[T] + self.eps) # (N, 1, 1, 1)
            else:
                vars_T  = (ddpm.sqrt_one_minus_alphas_bar[T-1] * ddpm.bw_std[T])**2 # (N, 1, 1, 1)
                weight = 0.5 * ddpm.square_betas[T] * ddpm.alphas_bar[T-1] / (vars_T * ddpm.square_one_minus_alphas_bar[T] + self.eps) # (N, 1, 1, 1)
                
            recon_loss = self.reconstruction_method(X_pred, X_input, reduction='none')
            recon_loss = (recon_loss*weight.expand_as(X_input)).mean()
            # recon_loss = self.reconstruction_method(X_pred, X_input, weight=weight.expand_as(X_input)) 
        else:
            recon_loss = self.reconstruction_method(X_pred, X_input)
        return {
            'Loss': recon_loss, 
        }