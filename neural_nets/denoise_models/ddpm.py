from pathlib import Path
from typing import (
    Tuple, 
    Type,
    Optional
)

import cv2
import numpy as np

import torch
from torch import (
    Tensor,
    linspace,
    cumprod,
    sqrt,
    randn_like,
    vmap
)
from torch.nn import Module
import torch.nn.functional as F
from torchvision.transforms import functional as Ft
from torch.nn.modules.utils import _pair

import matplotlib.pyplot as plt

from neural_nets.denoise_models.unets import _UNetBase

from utils import other_utils


class DDPM:
    def __init__(
        self,
        model: Type[Module] | None = None,
        beta: float = 0.02,
        T: int = 100,
        device=None,
        **model_kwargs
    ):
        self.img_channels = model_kwargs["img_channels"]
        self.T = T
        self.device = device
        
        if model is not None:
            self.model = model
        else:
            self.model = _UNetBase(
                T=T,
                is_conditional=False,
                device=device,
                **model_kwargs
            )
            
        betas = linspace(1e-4, beta, steps=T, device=device) # T time step 1 -> T, but indexed 0 -> T-1 
        self.sqrt_betas = sqrt(betas)
        alphas = 1.0 - betas
        alphas_bar = cumprod(alphas, dim=0)
        one_minus_alphas = 1.0 - alphas
        one_minus_alphas_bar = 1.0 - alphas_bar
        sqrt_one_minus_alphas_bar = sqrt(one_minus_alphas_bar)
        
        # foward diffusion hyperparameters
        self.diff_mean_coef = sqrt(alphas_bar)
        self.diff_std = sqrt_one_minus_alphas_bar
        
        # forward diff batched function
        self.q_batched = vmap(
            self.q, 
            in_dims=0,
            out_dims=0, 
            randomness="different"
        )
        
        # reverse diffusion hyperparameters
        self.noise_coef = one_minus_alphas / sqrt_one_minus_alphas_bar
        self.inv_sqrt_alphas = sqrt(1.0 / alphas)
        self.beta_tilde_coef = 1.0 # this is different from the original paper, but in practice it works wells
        
        # sampling (reverse diff) batched function
        self.reverse_q_batched = vmap(
            self.reverse_q, 
            in_dims=0, 
            out_dims=0,
            randomness="different"
        )
        
    
    def q(
        self, 
        x_0: Tensor, 
        t: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward diffusion process

        Args:
            x_0 (Tensor): Initial latent
            t (int): Time step at t (shifted to the left, so that t = 0 means it is the step 1)

        Shape:
            x_0: [N, C, H, W] or [C, H, W]
            t: [1]
        
        Returns:
            Tensor [N, C, H, W] or [C, H, W]: x_t, which is the added noise version of x_0 after t steps
            Tensor [N, C, H, W] or [C, H, W]: the noise added at step t
        """
        t = t.int()
        noise = randn_like(x_0)
        x_t = self.diff_mean_coef[t]*x_0 + self.diff_std[t]*noise
        return (x_t, noise)
        
        
    @torch.no_grad    
    def reverse_q(
        self,
        x_t: Tensor, 
        t: Tensor,
        noise_predicted: Tensor 
    ) -> Tensor:
        """
        Reverse diffusion process. x_t-1 is sampled based on current x_t,
        current time step t and predicted noise to produce x_t

        Args:
            x_t (Tensor): current sample at time step t
            t (Tensor): Time step at t (shifted to the right, so that t = 0 means it is the step 1)
            noise_predicted (Tensor): the noise that produce x_t


        Shape:
            x_t: [N, C, H, W]
            t: [1]
            noise_predicted: [N, C, H, W]

        Returns:
            Tensor [N, C, H, W]: x_t-1, the previous sample at time step t 
        """
        t = t.int()
        mean_t_minus_1 = self.inv_sqrt_alphas[t] * (x_t - self.noise_coef[t]*noise_predicted)
        # if t == 0:
        #     return mean_t_minus_1
        # else:
        #     noise = randn_like(x_t)
        #     # return mean_t_minus_1 + self.beta_tilde_coef * self.sqrt_betas[t] * noise   
        #     return mean_t_minus_1 + self.beta_tilde_coef * self.sqrt_betas[t-1] * noise
        
        noise = randn_like(x_t)
        return mean_t_minus_1 + self.beta_tilde_coef * self.sqrt_betas[t] * noise
        
    
    @torch.no_grad
    def show_translative_image(
        self,
        img_size: int | Tuple[int, int] = 256,
        save_path: str = None,
    ):
        img_size = _pair(img_size)
        # Init noise to generate images from
        x_t = torch.randn((1, self.img_channels, *img_size), device=self.device) # [N, C, H, W]

        plt.figure(figsize=(10, 10))
        n_cols = 10
        hidden_rows = self.T / n_cols
        plot_number = 1
        
        # Go from T to 0 removing and adding noise until t = 0
        self.model.eval()
        for t in range(0, self.T)[::-1]:
            time = torch.full((1, 1), t, device=self.device).float()
            e_t = self.model(x_t, time)  # Predicted noise
            x_t = self.reverse_q(x_t, time[0], e_t)
            if t % hidden_rows == 0:
                ax = plt.subplot(1, n_cols+1, plot_number)
                ax.axis('off')
                other_utils.show_tensor_image_diff(x_t)
                plot_number += 1

        # save the image
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        #plot image
        plt.show()


class LatentDDPM(DDPM):
    def __init__(
        self,
        encoder: Type[Module],
        decoder: Type[Module],
        model: Type[Module] | None = None,
        conditional_domain_encoder: Type[Module] | None = None,
        is_conditional: bool = False,
        beta: float = 0.02,
        T: int = 100,
        device=None,
        **model_kwargs
    ):
        super().__init__(model, beta, T, device, **model_kwargs)
        
        self.encoder = encoder
        self.decoder = decoder
        if is_conditional and conditional_domain_encoder is None:
            conditional_domain_encoder = encoder
        self.conditional_domain_encoder = conditional_domain_encoder
        
        self.latent_channels = model_kwargs["img_channels"]
        self.T = T
        self.device = device
        self.is_conditional = is_conditional
        
        if model is not None:
            self.model = model
        else:
            self.model = _UNetBase(
                T=T,
                is_conditional=is_conditional,
                device=device,
                **model_kwargs
            )   
    
    
    @torch.inference_mode
    def sample(
        self, 
        latent_size: int | Tuple[int, int] | None = None,
        num_image: int = 1,
        condition: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward method is the sampling process

        Args:
            latent_size (int | Tuple[int, int]): size of latent tensor.
            condition (Optional[Tensor], optional): Conditional image. Defaults to None.

        Returns:
            Tensor [N, 3, H, W]: The synthesis image
        """
        # Init noise to generate images from
        latent_size = _pair(latent_size)
        x_t = torch.randn((num_image, self.img_channels, *latent_size), device=self.device)
              
        # Process the conditional input if provided
        if condition is not None:
            self.conditional_domain_encoder.eval()
            condition: Tensor = self.conditional_domain_encoder(condition) # [N, C, H, W]
            condition = condition.flatten(start_dim=2).transpose(1, 2) # [N, H*W, C]
            condition = {
                "condition": condition,
                "need_weights": False,
            }
        else:
            condition = {}
            
        for t in range(0, self.T)[::-1]:
            time = torch.full((1, ), t, device=self.device).float()
            e_t = self.model(x_t, time, **condition)  # Predicted noise
            x_t = self.reverse_q(x_t, time, e_t)
        
        self.decoder.eval()
        return self.decoder(x_t).detach().cpu()
        
    
    @torch.inference_mode
    def show_translative_image(
        self,
        latent_size: int | Tuple[int, int] | None = 32,
        condition: Optional[Tensor] = None,
        save_path: str = None,
    ):
        # Init noise to generate images from
        if latent_size is not None: 
            latent_size = _pair(latent_size) 
        elif condition is None:
            raise ValueError("If latent_size is not specified, condition is expected to be provided")
        else:
            assert condition.dim() == 3, f"Expected condition has 3 dimensions, got {condition.dim()}"
            latent_size = condition.shape[1:] # [H, W]
            latent_size = (latent_size[0] // 8, latent_size[1] // 8)
        x_t = torch.randn((1, self.latent_channels, *latent_size), device=self.device) # [1, C, H, W]
        
        # Process the conditional input if provided
        if self.is_conditional:
            assert condition is not None, f"conditional is required, but not provided"
            self.conditional_domain_encoder.eval()
            condition = condition.unsqueeze(0) # [1, C, H, W]            
            condition = self.conditional_domain_encoder(condition) # [1, C, H, W]
            condition = condition.flatten(start_dim=2).transpose(1, 2).contiguous() # [1, H*W, C]
        else:
            condition = None

        plt.figure(figsize=(10, 10))
        n_cols = 10
        hidden_rows = self.T / n_cols
        plot_number = 1
        
        # Go from T to 0 removing and adding noise until t = 0
        self.decoder.eval()
        self.model.eval()
        for t in range(0, self.T)[::-1]:
            time = torch.full((1, 1), t, device=self.device).float()
            e_t = self.model(x_t, time, condition)  # Predicted noise
            x_t = self.reverse_q(x_t, time[0], e_t)
            if t % hidden_rows == 0:
                ax = plt.subplot(1, n_cols+1, plot_number)
                ax.axis('off')
                img: Tensor = self.decoder(x_t)
                other_utils.show_tensor_image_latent_diff(img)
                plot_number += 1

        # save the image
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        #plot image
        plt.show()
     

    def preprocess_mask2img(
        self,
        mask_path: str | Path,
        size: int | Tuple[int, int] = (256, 256)
    ) -> Tensor:
        
        def add_padding(image):
            h, w = image.shape[:2]

            if h > w:
                # Calculate the padding for width
                pad_width = (h - w) // 2
                pad = ((0, 0), (pad_width, pad_width))
            else:
                # Calculate the padding for height
                pad_height = (w - h) // 2
                pad = ((pad_height, pad_height), (0, 0))

            if len(image.shape) == 3:
                pad += ((0, 0), )    
            padded_image = np.pad(image, pad, mode='constant', constant_values=0)
            return padded_image
        
         # read images
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # padding
        mask = add_padding(mask)
        
        # resize images
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        
        # scale image values to [0, 1]
        mask = mask/255.0
        mask = mask.transpose(2, 0, 1)
        mask = np.ascontiguousarray(mask, dtype=np.float32)
        
        
        
        # convert to pytorch tensor
        mask = torch.from_numpy(mask)

        # scale input
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        mask = Ft.normalize(mask, mean=mean, std=std)
        
        return mask



# classifier free approach - sample context noise and avg noise
@torch.no_grad()
def sample_w(
    model, ddpm, input_size, T, c, device, w_tests=None, store_freq=10
):
    if w_tests is None:
        w_tests = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    # Preprase "grid of samples" with w for rows and c for columns
    n_samples = len(w_tests) * len(c)

    # One w for each c
    w = torch.tensor(w_tests).float().repeat_interleave(len(c))
    w = w[:, None, None, None].to(device)  # Make w broadcastable
    x_t = torch.randn(n_samples, *input_size).to(device)

    # One c for each w
    c = c.repeat(len(w_tests), 1)

    # Double the batch
    c = c.repeat(2, 1)

    # Don't drop context at test time
    c_mask = torch.ones_like(c).to(device)
    c_mask[n_samples:] = 0.0

    x_t_store = []
    for i in range(0, T)[::-1]:
        # Duplicate t for each sample
        t = torch.tensor([i]).to(device)
        t = t.repeat(n_samples, 1, 1, 1)

        # Double the batch
        x_t = x_t.repeat(2, 1, 1, 1)
        t = t.repeat(2, 1, 1, 1)

        # Find weighted noise
        e_t = model(x_t, t, c, c_mask)
        e_t_keep_c = e_t[:n_samples]
        e_t_drop_c = e_t[n_samples:]
        e_t = (1 + w) * e_t_keep_c - w * e_t_drop_c

        # Deduplicate batch for reverse diffusion
        x_t = x_t[:n_samples]
        t = t[:n_samples]
        x_t = ddpm.reverse_q(x_t, t, e_t)

        # Store values for animation
        if i % store_freq == 0 or i == T or i < 10:
            x_t_store.append(x_t)

    x_t_store = torch.stack(x_t_store)
    return x_t, x_t_store